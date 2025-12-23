"""
torchrun --nproc_per_node=8 compute_dual_kl_qwen.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B --dataset aime24 --aime_split train --max_samples 8 --max_new_tokens 2048 --dtype bf16 --ddp --output_json output-computekl-base/dual_kl_metrics.json --plot_dir output-computekl-base/entropy_plots --do_sample --temperature 0.7 --top_p 0.95


torchrun --nproc_per_node=8 compute_dual_kl_qwen.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B --dataset aime24 --aime_split train --max_samples 8 --max_new_tokens 2048 --dtype bf16 --ddp --output_json output-computekl-opd/dual_kl_metrics.json --plot_dir output-computekl-opd/entropy_plots --do_sample --temperature 0.7 --top_p 0.95 \
    --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-1.7b-32b-deepmath-long/step-500


torchrun --nproc_per_node=8 compute_dual_kl_qwen.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B --dataset aime24 --aime_split train --max_samples 8 --max_new_tokens 2048 --dtype bf16 --ddp --output_json output-computekl-dklr1f0.5/dual_kl_metrics.json --plot_dir output-computekl-dklr1f0.5/entropy_plots --do_sample --temperature 0.7 --top_p 0.95 \
    --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f0.5/step-500
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


try:
    from datasets import load_dataset  # type: ignore
    _HAVE_DATASETS = True
except Exception:
    _HAVE_DATASETS = False

try:
    from peft import PeftModel  # type: ignore
    _HAVE_PEFT = True
except Exception:
    _HAVE_PEFT = False

try:
    from tqdm.auto import tqdm  # type: ignore
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


@dataclass
class Args:
    teacher_model: str
    student_model: str
    student_lora: Optional[str]
    dataset: str
    aime_split: str
    jsonl_file: Optional[str]
    input_key: Optional[str]
    max_samples: Optional[int]
    max_prompt_tokens: Optional[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    eos_token: Optional[str]
    dtype: str
    device_map: str
    ddp: bool
    seed: int
    output_json: Optional[str]
    plot_dir: Optional[str]


def set_seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def ddp_init_if_needed(enable: bool) -> Tuple[bool, int, int, int, torch.device]:
    import os
    if enable and torch.cuda.is_available():
        import torch.distributed as dist
        if not dist.is_initialized():
            backend = "nccl"
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world, local_rank, device
    # Fallback single process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def ddp_barrier(ddp_on: bool) -> None:
    if ddp_on:
        import torch.distributed as dist
        dist.barrier()


def ddp_all_gather_floats(ddp_on: bool, device: torch.device, values: List[float]) -> List[float]:
    if not ddp_on:
        return list(values)
    import torch.distributed as dist
    t = torch.as_tensor(values, dtype=torch.float32, device=device)
    n_local = torch.tensor([t.numel()], dtype=torch.long, device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, n_local)
    sizes_cpu = [int(s.item()) for s in sizes]
    max_n = max(sizes_cpu)
    if t.numel() < max_n:
        pad = torch.zeros(max_n - t.numel(), dtype=t.dtype, device=device)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t
    gather_bufs = [torch.zeros(max_n, dtype=t.dtype, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_bufs, t_pad)
    out: List[float] = []
    for buf, n in zip(gather_bufs, sizes_cpu):
        out.extend(buf[:n].tolist())
    return out


def ddp_all_reduce_sum_int(ddp_on: bool, device: torch.device, value: int) -> int:
    if not ddp_on:
        return int(value)
    import torch.distributed as dist
    t = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def ddp_all_reduce_sum_float(ddp_on: bool, device: torch.device, value: float) -> float:
    if not ddp_on:
        return float(value)
    import torch.distributed as dist
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def load_models_and_tokenizer(args: Args, device: torch.device, ddp_on: bool) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(args.dtype.lower(), torch.bfloat16)

    # When DDP is on, put entire model on the local GPU; avoid device_map to prevent multi-GPU contention.
    if ddp_on:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(device)
        student_base = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(device)
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch_dtype,
            device_map=None if args.device_map == "none" else args.device_map,
        )
        student_base = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            torch_dtype=torch_dtype,
            device_map=None if args.device_map == "none" else args.device_map,
        )
    if args.student_lora:
        if not _HAVE_PEFT:
            raise RuntimeError("peft not installed; cannot load LoRA adapter. pip install peft")
        student = PeftModel.from_pretrained(student_base, args.student_lora)
        if ddp_on:
            student = student.to(device)
    else:
        student = student_base

    # Use student tokenizer for both to ensure consistent chat template
    tok = AutoTokenizer.from_pretrained(args.student_model)
    ensure_pad_token(tok)

    teacher.eval()
    student.eval()
    return teacher, student, tok


def apply_chat_template(tok: PreTrainedTokenizerBase, question: str) -> str:
    # Prefer tokenizer's chat template if available
    try:
        if hasattr(tok, "apply_chat_template") and callable(tok.apply_chat_template):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    # Fallback: plain user prefix
    return f"User: {question}\nAssistant:"


def load_questions(args: Args) -> List[str]:
    if args.dataset.lower() in ("aime24", "aime", "aime_2024", "hf:aime_2024"):
        if not _HAVE_DATASETS:
            raise RuntimeError("datasets not installed; pip install datasets or provide --jsonl_file")
        ds = load_dataset("HuggingFaceH4/aime_2024", split=args.aime_split)
        # Try common keys
        def pick(row: Dict, keys: Sequence[str]) -> Optional[str]:
            for k in keys:
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return None

        q_keys = ["question", "problem", "prompt", "input"]
        out: List[str] = []
        for row in ds:  # type: ignore
            q = pick(row, q_keys) or ""
            if q:
                out.append(q)
        return out

    if args.jsonl_file:
        import json

        key = args.input_key or "question"
        out: List[str] = []
        with open(args.jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                q = row.get(key) or row.get("prompt") or row.get("input")
                if isinstance(q, str) and q.strip():
                    out.append(q)
        return out

    raise ValueError("Unsupported dataset source. Use --dataset aime24 or provide --jsonl_file")


@torch.no_grad()
def init_prefix(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompt_text: str
) -> Tuple[torch.Tensor, Dict]:
    dev = device_of(model)
    enc = tok(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(dev)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    pkv = {"past_key_values": out.past_key_values, "attention_mask": attn_mask}
    return input_ids, pkv


@torch.no_grad()
def step_next(
    model: PreTrainedModel, last_token_id: torch.Tensor, pkv: Dict
) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
    # last_token_id: shape [1]
    dev = device_of(model)
    inp = last_token_id.view(1, 1).to(dev)
    out = model(input_ids=inp, use_cache=True, past_key_values=pkv.get("past_key_values"), attention_mask=pkv.get("attention_mask"))
    logits = out.logits[:, -1, :]
    pkv_new = {"past_key_values": out.past_key_values, "attention_mask": pkv.get("attention_mask")}
    return logits, pkv_new, inp.squeeze(0)


def sample_token(
    logits: torch.Tensor, tok: PreTrainedTokenizerBase, do_sample: bool, temperature: float, top_p: float
) -> int:
    logits = logits.to(torch.float32)
    if do_sample:
        scaled = logits / max(1e-6, float(temperature))
        probs = torch.softmax(scaled, dim=-1)
        if 0.0 < top_p < 1.0:
            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_probs, num_samples=1)
            token_id = sorted_idx.gather(-1, idx).item()
        else:
            token_id = torch.multinomial(probs, num_samples=1).item()
        return int(token_id)
    else:
        return int(torch.argmax(logits, dim=-1).item())


def kl_forward_reverse(
    logp_a: torch.Tensor, logp_b: torch.Tensor, forward: bool
) -> float:
    # logp_*: shape [V]
    logp_a = logp_a.to(torch.float32)
    logp_b = logp_b.to(torch.float32)
    if forward:
        # KL(Pa || Pb) = sum Pa (log Pa - log Pb)
        pa = torch.exp(logp_a)
        return float(torch.sum(pa * (logp_a - logp_b)).item())
    else:
        # KL(Pb || Pa)
        pb = torch.exp(logp_b)
        return float(torch.sum(pb * (logp_b - logp_a)).item())


def entropy_from_logp(logp: torch.Tensor) -> float:
    p = torch.exp(logp.to(torch.float32))
    h = -torch.sum(p * logp.to(torch.float32)).item()
    return float(h)


def follow_path_measure(
    path_model: PreTrainedModel,
    other_model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    forward_along_path: bool,
) -> Tuple[List[float], List[float], List[float]]:
    # Returns: (per-step KL along this path, entropies of PATH model, entropies of OTHER model) per step
    ids_path, pkv_path = init_prefix(path_model, tok, prompt_text)
    ids_other, pkv_other = init_prefix(other_model, tok, prompt_text)

    last_token = ids_path[0, -1]
    entropies_path: List[float] = []
    entropies_other: List[float] = []
    kls: List[float] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Step both models with the same last token (path token)
            logits_path, pkv_path, _ = step_next(path_model, last_token, pkv_path)
            logits_other, pkv_other, _ = step_next(other_model, last_token, pkv_other)

            logp_path = torch.log_softmax(logits_path.squeeze(0), dim=-1)
            logp_other = torch.log_softmax(logits_other.squeeze(0), dim=-1)

            # KL along this path
            kl_val = kl_forward_reverse(logp_path, logp_other, forward=forward_along_path)
            kls.append(kl_val)
            entropies_path.append(entropy_from_logp(logp_path))
            entropies_other.append(entropy_from_logp(logp_other))

            # Choose next token from PATH model
            next_id = sample_token(logits_path.squeeze(0), tok, do_sample=do_sample, temperature=temperature, top_p=top_p)
            last_token = torch.tensor(next_id, dtype=ids_path.dtype, device=device_of(path_model))
            if eos_token_id is not None and int(next_id) == int(eos_token_id):
                break

    return kls, entropies_path, entropies_other


def describe_distribution(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0}
    import numpy as np  # type: ignore

    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, float | int] = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute forward/reverse KL between teacher/student across paths (AIME24)")
    ap.add_argument("--teacher_model", type=str, required=True, help="Teacher model id or path (e.g., Qwen/Qwen3-32B-Instruct)")
    ap.add_argument("--student_model", type=str, required=True, help="Student model id or path (e.g., Qwen/Qwen3-1.7B-Instruct)")
    ap.add_argument("--student_lora", type=str, default=None, help="Optional path to LoRA adapter for the student")
    ap.add_argument("--dataset", type=str, default="aime24", help="Dataset selector: aime24 or custom")
    ap.add_argument("--aime_split", type=str, default="test", help="HuggingFaceH4/aime_2024 split")
    ap.add_argument("--jsonl_file", type=str, default=None, help="Local JSONL with a 'question' field if not using AIME")
    ap.add_argument("--input_key", type=str, default=None, help="Key name for question in JSONL (default: question)")
    ap.add_argument("--max_samples", type=int, default=20, help="Max number of questions to evaluate")
    ap.add_argument("--max_prompt_tokens", type=int, default=None, help="Truncate prompt to at most this many tokens")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--do_sample", action="store_true", help="Use sampling for path following (default: greedy)")
    ap.add_argument("--eos_token", type=str, default=None, help="Optional EOS string to stop generation early")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]) 
    ap.add_argument("--device_map", type=str, default="auto", help="Transformers device_map (auto|cpu|cuda|balanced|none|ddp)")
    ap.add_argument("--ddp", action="store_true", help="Enable multi-GPU data-parallel with torch.distributed (use torchrun)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics JSON")
    ap.add_argument("--plot_dir", type=str, default="kl_plots", help="Directory to save entropy histograms")
    a = ap.parse_args()
    args = Args(
        teacher_model=a.teacher_model,
        student_model=a.student_model,
        student_lora=a.student_lora,
        dataset=a.dataset,
        aime_split=a.aime_split,
        jsonl_file=a.jsonl_file,
        input_key=a.input_key,
        max_samples=a.max_samples,
        max_prompt_tokens=a.max_prompt_tokens,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        do_sample=bool(a.do_sample),
        eos_token=a.eos_token,
        dtype=a.dtype,
        device_map=a.device_map,
        ddp=bool(a.ddp or a.device_map == "ddp"),
        seed=a.seed,
        output_json=a.output_json,
        plot_dir=a.plot_dir,
    )

    set_seed_all(args.seed)

    ddp_on, rank, world, local_rank, device = ddp_init_if_needed(args.ddp)
    if ddp_on and args.device_map != "ddp":
        # If user forgot to set device_map for DDP, we still force single-device per rank
        pass

    teacher, student, tok = load_models_and_tokenizer(args, device=device, ddp_on=ddp_on)
    eos_token_id = None
    if args.eos_token is not None:
        try:
            eos_token_id = tok.convert_tokens_to_ids(args.eos_token)
        except Exception:
            eos_token_id = tok.eos_token_id
    else:
        eos_token_id = tok.eos_token_id

    questions = load_questions(args)
    if args.max_samples is not None:
        questions = questions[: max(0, int(args.max_samples))]

    prompts: List[str] = []
    for q in questions:
        txt = q
        if args.max_prompt_tokens is not None and args.max_prompt_tokens > 0:
            # quick truncation by tokens
            ids = tok.encode(q)
            if len(ids) > args.max_prompt_tokens:
                txt = tok.decode(ids[: args.max_prompt_tokens])
        prompts.append(apply_chat_template(tok, txt))

    fkl_values: List[float] = []  # teacher path KL(teacher || student)
    rkl_values: List[float] = []  # student path KL(student || teacher)
    ent_teacher: List[float] = []  # entropies along teacher path (teacher distro)
    ent_student: List[float] = []  # entropies along student path (student distro)

    # Shard prompts across ranks for DDP
    idxs = list(range(len(prompts)))
    if ddp_on:
        shard_prompts = [prompts[i] for i in idxs if (i % world) == rank]
    else:
        shard_prompts = prompts

    # Progress bar (rank 0 only in DDP to avoid clutter)
    use_bar = _HAVE_TQDM and (not ddp_on or rank == 0)
    pbar = tqdm(total=len(shard_prompts), desc=f"rank {rank}/{world} prompts", dynamic_ncols=True, leave=True) if use_bar else None

    # Track per-path output lengths (number of generated tokens)
    teacher_steps_sum_local = 0
    student_steps_sum_local = 0
    teacher_prompts_local = 0
    student_prompts_local = 0

    # Track per-path output lengths (number of generated tokens)
    teacher_steps_sum_local = 0
    student_steps_sum_local = 0
    teacher_prompts_local = 0
    student_prompts_local = 0

    # Track student's entropy along teacher path
    ent_student_on_teacher_local: List[float] = []

    for i, p in enumerate(shard_prompts):
        # Teacher path -> forward KL
        kls_t, ents_t, ents_t_other = follow_path_measure(
            path_model=teacher,
            other_model=student,
            tok=tok,
            prompt_text=p,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            forward_along_path=True,
        )
        fkl_values.extend(kls_t)
        ent_teacher.extend(ents_t)
        ent_student_on_teacher_local.extend(ents_t_other)
        teacher_steps_sum_local += len(kls_t)
        teacher_prompts_local += 1

        # Student path -> reverse KL
        kls_s, ents_s, _ = follow_path_measure(
            path_model=student,
            other_model=teacher,
            tok=tok,
            prompt_text=p,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            forward_along_path=False,
        )
        rkl_values.extend(kls_s)
        ent_student.extend(ents_s)
        student_steps_sum_local += len(kls_s)
        student_prompts_local += 1

        if pbar is not None:
            pbar.update(1)
        elif (i + 1) % 1 == 0 and (not ddp_on or rank == 0):
            print(f"[progress] processed {i+1}/{len(shard_prompts)} prompts (rank {rank}/{world}); tokens so far: fKL={len(fkl_values)}, rKL={len(rkl_values)}")

    def avg(vals: List[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    if pbar is not None:
        pbar.close()

    # DDP: reduce sums and counts for KL means; gather entropies for histogram/stats
    total_fkl_sum = ddp_all_reduce_sum_float(ddp_on, device, float(sum(fkl_values)))
    total_fkl_count = ddp_all_reduce_sum_int(ddp_on, device, int(len(fkl_values)))
    total_rkl_sum = ddp_all_reduce_sum_float(ddp_on, device, float(sum(rkl_values)))
    total_rkl_count = ddp_all_reduce_sum_int(ddp_on, device, int(len(rkl_values)))

    total_teacher_steps = ddp_all_reduce_sum_int(ddp_on, device, int(teacher_steps_sum_local))
    total_teacher_prompts = ddp_all_reduce_sum_int(ddp_on, device, int(teacher_prompts_local))
    total_student_steps = ddp_all_reduce_sum_int(ddp_on, device, int(student_steps_sum_local))
    total_student_prompts = ddp_all_reduce_sum_int(ddp_on, device, int(student_prompts_local))

    all_ent_teacher = ddp_all_gather_floats(ddp_on, device, ent_teacher)
    all_ent_student = ddp_all_gather_floats(ddp_on, device, ent_student)
    all_ent_student_on_teacher = ddp_all_gather_floats(ddp_on, device, ent_student_on_teacher_local)

    if not ddp_on or rank == 0:
        report = {
            "forward_KL_teacher_path_mean": (total_fkl_sum / max(1, total_fkl_count)),
            "reverse_KL_student_path_mean": (total_rkl_sum / max(1, total_rkl_count)),
            "teacher_entropy_mean": avg(all_ent_teacher),
            "student_entropy_mean": avg(all_ent_student),
            "student_on_teacher_entropy_mean": avg(all_ent_student_on_teacher),
            "teacher_entropy_stats": describe_distribution(all_ent_teacher),
            "student_entropy_stats": describe_distribution(all_ent_student),
            "student_on_teacher_entropy_stats": describe_distribution(all_ent_student_on_teacher),
            "teacher_path_avg_output_len": (float(total_teacher_steps) / max(1, int(total_teacher_prompts))),
            "student_path_avg_output_len": (float(total_student_steps) / max(1, int(total_student_prompts))),
            "counts": {
                "forward_kl_tokens": int(total_fkl_count),
                "reverse_kl_tokens": int(total_rkl_count),
                "teacher_prompts": int(total_teacher_prompts),
                "student_prompts": int(total_student_prompts),
            },
        }

        print("\n==== Dual-Path KL and Entropy Summary ====")
        print(json.dumps(report, ensure_ascii=False, indent=2))

        # Plot histograms
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
            try:
                import seaborn as sns  # type: ignore
                sns.set_theme(style="whitegrid")
            except Exception:
                plt.style.use("seaborn-v0_8-darkgrid")

            os.makedirs(args.plot_dir or "kl_plots", exist_ok=True)
            # Overlay plot (Teacher path vs Student path)
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = 40
            if (len(all_ent_teacher) + len(all_ent_student)) > 0:
                vmin = float(min(all_ent_teacher + all_ent_student))
                vmax = float(max(all_ent_teacher + all_ent_student))
                if math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin:
                    import numpy as np  # type: ignore
                    bin_edges = np.linspace(vmin, vmax, bins + 1)
                else:
                    bin_edges = bins
            else:
                bin_edges = bins
            ax.hist(all_ent_teacher, bins=bin_edges, alpha=0.55, color="#1f77b4", label="Teacher entropy")
            ax.hist(all_ent_student, bins=bin_edges, alpha=0.55, color="#ff7f0e", label="Student entropy")
            ax.set_title("Entropy Distribution (Teacher vs Student)")
            ax.set_xlabel("Entropy (nats)")
            ax.set_ylabel("Count")
            ax.legend()
            overlay_path = os.path.join(args.plot_dir or "kl_plots", "entropy_overlay.png")
            fig.tight_layout()
            fig.savefig(overlay_path, dpi=150)
            plt.close(fig)

            # Overlay proportion histogram (normalized per histogram)
            figp, axp = plt.subplots(figsize=(8, 5))
            if isinstance(bin_edges, int):
                be = bin_edges
            else:
                be = bin_edges
            wt = np.ones(len(all_ent_teacher), dtype=float) / max(1, len(all_ent_teacher))
            ws = np.ones(len(all_ent_student), dtype=float) / max(1, len(all_ent_student))
            axp.hist(all_ent_teacher, bins=be, weights=wt, alpha=0.55, color="#1f77b4", label="Teacher entropy")
            axp.hist(all_ent_student, bins=be, weights=ws, alpha=0.55, color="#ff7f0e", label="Student entropy")
            axp.set_title("Entropy Distribution (Proportion)")
            axp.set_xlabel("Entropy (nats)")
            axp.set_ylabel("Proportion")
            axp.legend()
            prop_path = os.path.join(args.plot_dir or "kl_plots", "entropy_overlay_proportion.png")
            figp.tight_layout()
            figp.savefig(prop_path, dpi=150)
            plt.close(figp)

            # Overlay plots specifically on teacher path: Teacher vs Student-on-Teacher
            fig_tpt, ax_tpt = plt.subplots(figsize=(8, 5))
            if (len(all_ent_teacher) + len(all_ent_student_on_teacher)) > 0:
                vmin_tp = float(min(all_ent_teacher + all_ent_student_on_teacher))
                vmax_tp = float(max(all_ent_teacher + all_ent_student_on_teacher))
                if math.isfinite(vmin_tp) and math.isfinite(vmax_tp) and vmax_tp > vmin_tp:
                    bin_edges_tp = np.linspace(vmin_tp, vmax_tp, bins + 1)
                else:
                    bin_edges_tp = bins
            else:
                bin_edges_tp = bins
            ax_tpt.hist(all_ent_teacher, bins=bin_edges_tp, alpha=0.55, color="#1f77b4", label="Teacher (teacher path)")
            ax_tpt.hist(all_ent_student_on_teacher, bins=bin_edges_tp, alpha=0.55, color="#2ca02c", label="Student on teacher path")
            ax_tpt.set_title("Entropy on Teacher Path (Count)")
            ax_tpt.set_xlabel("Entropy (nats)")
            ax_tpt.set_ylabel("Count")
            ax_tpt.legend()
            tpt_overlay = os.path.join(args.plot_dir or "kl_plots", "entropy_teacherpath_overlay.png")
            fig_tpt.tight_layout()
            fig_tpt.savefig(tpt_overlay, dpi=150)
            plt.close(fig_tpt)

            fig_tpp, ax_tpp = plt.subplots(figsize=(8, 5))
            be_tp = bin_edges_tp if not isinstance(bin_edges_tp, int) else bin_edges_tp
            wt_tp_t = np.ones(len(all_ent_teacher), dtype=float) / max(1, len(all_ent_teacher))
            wt_tp_s = np.ones(len(all_ent_student_on_teacher), dtype=float) / max(1, len(all_ent_student_on_teacher))
            ax_tpp.hist(all_ent_teacher, bins=be_tp, weights=wt_tp_t, alpha=0.55, color="#1f77b4", label="Teacher (teacher path)")
            ax_tpp.hist(all_ent_student_on_teacher, bins=be_tp, weights=wt_tp_s, alpha=0.55, color="#2ca02c", label="Student on teacher path")
            ax_tpp.set_title("Entropy on Teacher Path (Proportion)")
            ax_tpp.set_xlabel("Entropy (nats)")
            ax_tpp.set_ylabel("Proportion")
            ax_tpp.legend()
            tpt_prop = os.path.join(args.plot_dir or "kl_plots", "entropy_teacherpath_overlay_proportion.png")
            fig_tpp.tight_layout()
            fig_tpp.savefig(tpt_prop, dpi=150)
            plt.close(fig_tpp)

            # Individual plots
            def plot_single(vals: List[float], title: str, color: str, fname: str) -> None:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.hist(vals, bins=bin_edges, alpha=0.8, color=color)
                ax2.set_title(title)
                ax2.set_xlabel("Entropy (nats)")
                ax2.set_ylabel("Count")
                fig2.tight_layout()
                fig2.savefig(os.path.join(args.plot_dir or "kl_plots", fname), dpi=150)
                plt.close(fig2)

            plot_single(all_ent_teacher, "Entropy Distribution - Teacher Path", "#1f77b4", "entropy_teacher.png")
            plot_single(all_ent_student, "Entropy Distribution - Student Path", "#ff7f0e", "entropy_student.png")
            plot_single(all_ent_student_on_teacher, "Entropy - Student on Teacher Path", "#2ca02c", "entropy_student_on_teacher.png")
            print(f"Saved entropy histograms to {args.plot_dir}")
        except Exception as e:
            print(f"[warn] Failed to render histograms: {e}")

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                f.write(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"Saved metrics to {args.output_json}")

    ddp_barrier(ddp_on)


if __name__ == "__main__":
    main()
