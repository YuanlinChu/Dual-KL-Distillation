"""
Example usages:

torchrun --nproc_per_node=8 compute_single_entropy_qwen.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --output_json output-Qwen3-1.7B-base/entropy_metrics.json \
  --plot_dir output-Qwen3-1.7B-base/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95

torchrun --nproc_per_node=8 compute_single_entropy_qwen.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-200 \
  --output_json output-Qwen3-1.7B-dklr1f1posdecay/entropy_metrics.json \
  --plot_dir output-Qwen3-1.7B-dklr1f1posdecay/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95

torchrun --nproc_per_node=8 compute_single_entropy_qwen.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --output_json output-DS-7B-base-t0/entropy_metrics.json \
  --plot_dir output-DS-7B-base-t0/entropy_plots \
  --do_sample --temperature 0.0 --top_p 0.95

torchrun --nproc_per_node=8 compute_single_entropy_qwen.py \
  --model Qwen/Qwen2.5-Math-1.5B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --output_json output-Qwen2.5-Math-1.5B-base/entropy_metrics.json \
  --plot_dir output-Qwen2.5-Math-1.5B-base/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95

torchrun --nproc_per_node=8 compute_single_entropy_qwen.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --output_json output-DS-1.5B-base-t0/entropy_metrics.json \
  --plot_dir output-DS-1.5B-base-t0/entropy_plots \
  --do_sample --temperature 0.0 --top_p 0.95

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 compute_single_entropy_qwen2.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --output_json output-Qwen3-1.7B-base2/entropy_metrics.json \
  --plot_dir output-Qwen3-1.7B-base2/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 compute_single_entropy_qwen2.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-200\
  --output_json output-Qwen3-1.7B-dklr1f1posdecay2/entropy_metrics.json \
  --plot_dir output-Qwen3-1.7B-dklr1f1posdecay2/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 compute_single_entropy_qwen2.py \
  --model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train \
  --max_samples 8 --max_new_tokens 8196 --dtype bf16 --ddp \
  --lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-1.7b-32b-deepmath-long/step-400\
  --output_json output-Qwen3-1.7B-opd400/entropy_metrics.json \
  --plot_dir output-Qwen3-1.7B-opd400/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95
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
    model: str
    lora: Optional[str]
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
    ignore_eos: bool
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


def load_model_and_tokenizer(args: Args, device: torch.device, ddp_on: bool) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(args.dtype.lower(), torch.bfloat16)

    if ddp_on:
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None if args.device_map == "none" else args.device_map,
        )

    if args.lora:
        if not _HAVE_PEFT:
            raise RuntimeError("peft not installed; cannot load LoRA adapter. pip install peft")
        model = PeftModel.from_pretrained(base, args.lora)
        if ddp_on:
            model = model.to(device)
    else:
        model = base

    tok = AutoTokenizer.from_pretrained(args.model)
    ensure_pad_token(tok)

    model.eval()
    return model, tok


def apply_chat_template(tok: PreTrainedTokenizerBase, question: str) -> str:
    try:
        if hasattr(tok, "apply_chat_template") and callable(tok.apply_chat_template):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return f"User: {question}\nAssistant:"


def load_questions(args: Args) -> List[str]:
    if args.dataset.lower() in ("aime24", "aime", "aime_2024", "hf:aime_2024"):
        if not _HAVE_DATASETS:
            raise RuntimeError("datasets not installed; pip install datasets or provide --jsonl_file")
        ds = load_dataset("HuggingFaceH4/aime_2024", split=args.aime_split)
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
    dev = device_of(model)
    inp = last_token_id.view(1, 1).to(dev)
    # 正确扩展 attention_mask（在过去长度基础上为新 token 追加 1）
    attn_mask = pkv.get("attention_mask")
    if attn_mask is not None:
        ones = torch.ones((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([attn_mask, ones], dim=1)
    out = model(
        input_ids=inp,
        use_cache=True,
        past_key_values=pkv.get("past_key_values"),
        attention_mask=attn_mask,
    )
    logits = out.logits[:, -1, :]
    pkv_new = {"past_key_values": out.past_key_values, "attention_mask": attn_mask}
    return logits, pkv_new, inp.squeeze(0)


def sample_token(
    logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float
) -> int:
    logits = logits.to(torch.float32)
    if do_sample:
        scaled = logits / max(1e-6, float(temperature))
        probs = torch.softmax(scaled, dim=-1)
        if 0.0 < top_p < 1.0:
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


def entropy_from_logp(logp: torch.Tensor) -> float:
    p = torch.exp(logp.to(torch.float32))
    h = -torch.sum(p * logp.to(torch.float32)).item()
    return float(h)


def follow_and_measure_entropy(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_ids: Optional[Sequence[int]],
    ignore_eos: bool,
) -> List[float]:
    ids, pkv = init_prefix(model, tok, prompt_text)
    last_token = ids[0, -1]
    entropies: List[float] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, pkv, _ = step_next(model, last_token, pkv)
            logp = torch.log_softmax(logits.squeeze(0), dim=-1)
            entropies.append(entropy_from_logp(logp))
            next_id = sample_token(logits.squeeze(0), do_sample=do_sample, temperature=temperature, top_p=top_p)
            last_token = torch.tensor(next_id, dtype=ids.dtype, device=device_of(model))
            if (not ignore_eos) and eos_token_ids is not None and int(next_id) in set(int(x) for x in eos_token_ids):
                break
    return entropies


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
    ap = argparse.ArgumentParser(description="Compute entropy distribution for a single Qwen model over generated tokens")
    ap.add_argument("--model", type=str, required=True, help="Model id or path (e.g., Qwen/Qwen3-1.7B-Instruct)")
    ap.add_argument("--lora", type=str, default=None, help="Optional path to LoRA adapter")
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
    ap.add_argument("--ignore_eos", action="store_true", help="Do not stop at EOS; always generate up to max_new_tokens")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]) 
    ap.add_argument("--device_map", type=str, default="auto", help="Transformers device_map (auto|cpu|cuda|balanced|none|ddp)")
    ap.add_argument("--ddp", action="store_true", help="Enable multi-GPU data-parallel with torch.distributed (use torchrun)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics JSON")
    ap.add_argument("--plot_dir", type=str, default="entropy_plots", help="Directory to save entropy histograms")
    a = ap.parse_args()

    args = Args(
        model=a.model,
        lora=a.lora,
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
        ignore_eos=bool(a.ignore_eos),
        dtype=a.dtype,
        device_map=a.device_map,
        ddp=bool(a.ddp or a.device_map == "ddp"),
        seed=a.seed,
        output_json=a.output_json,
        plot_dir=a.plot_dir,
    )

    set_seed_all(args.seed)

    ddp_on, rank, world, local_rank, device = ddp_init_if_needed(args.ddp)

    model, tok = load_model_and_tokenizer(args, device=device, ddp_on=ddp_on)
    # 统一处理单个或多个 EOS id（优先使用模型 generation_config）
    eos_token_ids: Optional[Sequence[int]] = None
    try:
        gen_eos = getattr(model.generation_config, "eos_token_id", None)
    except Exception:
        gen_eos = None
    if args.eos_token is not None:
        try:
            conv = tok.convert_tokens_to_ids(args.eos_token)
            if isinstance(conv, (list, tuple)):
                eos_token_ids = list(int(x) for x in conv)
            else:
                eos_token_ids = [int(conv)]
        except Exception:
            pass
    if eos_token_ids is None:
        if gen_eos is not None:
            if isinstance(gen_eos, (list, tuple)):
                eos_token_ids = list(int(x) for x in gen_eos)
            else:
                eos_token_ids = [int(gen_eos)]
        else:
            eos = getattr(tok, "eos_token_id", None)
            if eos is not None:
                if isinstance(eos, (list, tuple)):
                    eos_token_ids = list(int(x) for x in eos)
                else:
                    eos_token_ids = [int(eos)]

    questions = load_questions(args)
    if args.max_samples is not None:
        questions = questions[: max(0, int(args.max_samples))]

    prompts: List[str] = []
    for q in questions:
        txt = q
        if args.max_prompt_tokens is not None and args.max_prompt_tokens > 0:
            ids = tok.encode(q)
            if len(ids) > args.max_prompt_tokens:
                txt = tok.decode(ids[: args.max_prompt_tokens])
        prompts.append(apply_chat_template(tok, txt))

    # DDP sharding
    idxs = list(range(len(prompts)))
    if ddp_on:
        shard_prompts = [prompts[i] for i in idxs if (i % world) == rank]
    else:
        shard_prompts = prompts

    use_bar = _HAVE_TQDM and (not ddp_on or rank == 0)
    pbar = tqdm(total=len(shard_prompts), desc=f"rank {rank}/{world} prompts", dynamic_ncols=True, leave=True) if use_bar else None

    entropies_local: List[float] = []
    steps_sum_local = 0
    prompts_local = 0

    for i, p in enumerate(shard_prompts):
        ents = follow_and_measure_entropy(
            model=model,
            tok=tok,
            prompt_text=p,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_ids=eos_token_ids,
            ignore_eos=bool(args.ignore_eos),
        )
        entropies_local.extend(ents)
        steps_sum_local += len(ents)
        prompts_local += 1
        if pbar is not None:
            pbar.update(1)
        elif (i + 1) % 1 == 0 and (not ddp_on or rank == 0):
            print(f"[progress] processed {i+1}/{len(shard_prompts)} prompts (rank {rank}/{world}); tokens so far: {len(entropies_local)}")

    if pbar is not None:
        pbar.close()

    # Reduce and gather
    total_steps = ddp_all_reduce_sum_int(ddp_on, device, int(steps_sum_local))
    total_prompts = ddp_all_reduce_sum_int(ddp_on, device, int(prompts_local))
    all_entropies = ddp_all_gather_floats(ddp_on, device, entropies_local)

    if not ddp_on or rank == 0:
        def avg(vals: List[float]) -> float:
            return float(sum(vals) / max(1, len(vals)))

        report = {
            "entropy_mean": avg(all_entropies),
            "entropy_stats": describe_distribution(all_entropies),
            "avg_output_len": (float(total_steps) / max(1, int(total_prompts))),
            "counts": {
                "tokens": int(len(all_entropies)),
                "prompts": int(total_prompts),
            },
        }

        print("\n==== Single-Model Entropy Summary ====")
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

            os.makedirs(args.plot_dir or "entropy_plots", exist_ok=True)

            bins = 40
            if len(all_entropies) > 0:
                vmin = float(min(all_entropies))
                vmax = float(max(all_entropies))
                if math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin:
                    bin_edges = np.linspace(vmin, vmax, bins + 1)
                else:
                    bin_edges = bins
            else:
                bin_edges = bins

            # Count histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(all_entropies, bins=bin_edges, alpha=0.8, color="#1f77b4")
            ax.set_title("Entropy Distribution (Count)")
            ax.set_xlabel("Entropy (nats)")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir or "entropy_plots", "entropy_single.png"), dpi=150)
            plt.close(fig)

            # Proportion histogram
            figp, axp = plt.subplots(figsize=(8, 5))
            be = bin_edges if not isinstance(bin_edges, int) else bin_edges
            weights = np.ones(len(all_entropies), dtype=float) / max(1, len(all_entropies))
            axp.hist(all_entropies, bins=be, weights=weights, alpha=0.8, color="#1f77b4")
            axp.set_title("Entropy Distribution (Proportion)")
            axp.set_xlabel("Entropy (nats)")
            axp.set_ylabel("Proportion")
            figp.tight_layout()
            figp.savefig(os.path.join(args.plot_dir or "entropy_plots", "entropy_single_proportion.png"), dpi=150)
            plt.close(figp)

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
