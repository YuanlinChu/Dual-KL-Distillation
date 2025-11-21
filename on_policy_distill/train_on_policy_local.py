from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from accelerate import Accelerator
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False
try:
    import deepspeed  # type: ignore
    DEEPSPEED_AVAILABLE = True
except Exception:
    DEEPSPEED_AVAILABLE = False
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


@dataclass
class Config:
    student_model: str
    teacher_model: str
    output_dir: str
    steps: int = 1000
    batch_size: int = 2  # prompts per batch (groups)
    group_size: int = 1  # rollouts per prompt
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    kl_coef: float = 1.0
    kl_discount: float = 0.0
    save_every: int = 100
    prompts_file: str | None = None
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    dtype: str = "bf16"  # bf16|fp16|fp32
    grad_accum: int = 1  # like num_substeps
    eval_every: int = 50
    dataset: str | None = None  # deepmath|tulu3|None
    max_prompt_tokens: int | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "online"  # online|offline|disabled
    # DeepSpeed options (teacher inference sharding)
    teacher_ds_zero3: bool = False
    teacher_ds_config: str | None = None
    # Micro-batching to reduce peak memory on student
    gen_micro_batch: int = 8
    lp_micro_batch: int = 8
    # Progress bar control
    progress: bool = True


def load_prompts(path: str | None) -> List[str]:
    if path is None or not os.path.exists(path):
        # Small built-in set for quick experiments
        return [
            "Explain the concept of entropy in simple terms.",
            "Write a short poem about the ocean.",
            "What are the pros and cons of unit testing?",
            "Summarize the key differences between HTTP/1.1 and HTTP/2.",
        ]
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_deepmath_prompts() -> List[str] | None:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        return [row["question"] for row in ds]  # type: ignore
    except Exception:
        return None


def load_tulu3_prompts() -> List[str] | None:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        prompts: List[str] = []
        for row in ds:  # type: ignore
            msgs = row["messages"]  # type: ignore
            for m in msgs:
                if m.get("role") == "user":
                    prompts.append(m.get("content", ""))
                    break
        return [p for p in prompts if p]
    except Exception:
        return None


def get_prompts(cfg: Config) -> List[str]:
    if cfg.prompts_file:
        return load_prompts(cfg.prompts_file)
    if cfg.dataset == "deepmath":
        p = load_deepmath_prompts()
        if p:
            return p
    if cfg.dataset == "tulu3":
        p = load_tulu3_prompts()
        if p:
            return p
    return load_prompts(None)


def truncate_by_tokens(tokenizer: PreTrainedTokenizerBase, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids)


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def device() -> torch.device:
    # Deprecated by Accelerator; kept for fallback
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def generate_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    micro_batch: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, List[int]]:
    # unwrap if DDP-wrapped
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    outs: List[torch.Tensor] = []
    plens: List[int] = []
    with torch.no_grad():
        iterator = range(0, len(prompts), max(1, micro_batch))
        bar = None
        if show_progress and _HAVE_TQDM:
            bar = tqdm(total=len(prompts), desc="gen", leave=False, dynamic_ncols=True)
        for i in iterator:
            chunk = prompts[i : i + max(1, micro_batch)]
            batch = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            batch = {k: v.to(next(model_for_gen.parameters()).device) for k, v in batch.items()}
            gen = model_for_gen.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outs.append(gen)
            plens.extend(batch["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist())
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()
    return torch.cat(outs, dim=0), plens


def sequence_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-token logprobs for targets (shifted) with shape [B, T-1]."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = nn.functional.log_softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:]
    logprobs = logprobs[:, :-1, :]
    gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return gathered


def discounted_future_sum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted future sum along last dimension.

    x: [..., T]
    returns: [..., T]
    """
    if gamma <= 0:
        return x
    y = torch.zeros_like(x)
    acc = torch.zeros_like(x[..., 0])
    for t in reversed(range(x.shape[-1])):
        acc = x[..., t] + gamma * acc
        y[..., t] = acc
    return y


def train_step(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    cfg: Config,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
) -> dict:
    # 1) Sample with the student (no grad)
    full_seqs, prompt_lengths = generate_continuations(
        student, tokenizer, prompts,
        cfg.max_new_tokens, cfg.temperature, cfg.top_p,
        cfg.gen_micro_batch,
        cfg.progress and accelerator.is_main_process,
    )

    attention_mask = full_seqs.ne(tokenizer.pad_token_id).long()

    # 2) Compute per-token logprobs for teacher (no grad) and student (with grad)
    teacher.eval()
    with torch.no_grad():
        parts_t: List[torch.Tensor] = []
        iterator_t = range(0, full_seqs.size(0), max(1, cfg.lp_micro_batch))
        bar_t = None
        if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
            bar_t = tqdm(total=full_seqs.size(0), desc="teacher lp", leave=False, dynamic_ncols=True)
        for i in iterator_t:
            sl = slice(i, i + max(1, cfg.lp_micro_batch))
            parts_t.append(sequence_logprobs(teacher, full_seqs[sl], attention_mask[sl]))
            if bar_t is not None:
                bar_t.update(full_seqs[sl].size(0))
        if bar_t is not None:
            bar_t.close()
        teach_lp = torch.cat(parts_t, dim=0)  # [B, T-1]

    student.train()
    # AMP for student forward/backward
    use_bf16 = cfg.dtype.lower() == "bf16" and torch.cuda.is_available()
    use_fp16 = cfg.dtype.lower() == "fp16" and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    if autocast_dtype is not None:
        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        # no-op context manager
        class _Null:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        ctx = _Null()

    with ctx:
        parts_s: List[torch.Tensor] = []
        iterator_s = range(0, full_seqs.size(0), max(1, cfg.lp_micro_batch))
        bar_s = None
        if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
            bar_s = tqdm(total=full_seqs.size(0), desc="student lp", leave=False, dynamic_ncols=True)
        for i in iterator_s:
            sl = slice(i, i + max(1, cfg.lp_micro_batch))
            parts_s.append(sequence_logprobs(student, full_seqs[sl], attention_mask[sl]))
            if bar_s is not None:
                bar_s.update(full_seqs[sl].size(0))
        if bar_s is not None:
            bar_s.close()
        stud_lp = torch.cat(parts_s, dim=0)  # [B, T-1]

    # Build mask for continuation tokens (exclude prompt tokens)
    # Position t in stud_lp corresponds to token at input_ids[:, t+1]
    B, Tp = stud_lp.shape
    cont_mask = torch.zeros_like(stud_lp, dtype=torch.bool)
    for i, L in enumerate(prompt_lengths):
        # tokens >= L belong to continuation; in stud_lp index space, that's t >= L-1
        start = max(L - 1, 0)
        cont_mask[i, start:] = True

    # 3) Compute advantages from reverse KL: A = -coef*(log p_s - log p_t)
    with torch.no_grad():
        delta = stud_lp.detach() - teach_lp  # [B, T-1]
        adv = -cfg.kl_coef * delta
        if cfg.kl_discount > 0:
            adv = discounted_future_sum(adv, cfg.kl_discount)
        # Optional variance reduction: center across valid tokens
        denom = cont_mask.sum().clamp_min(1)
        mean_adv = (adv.masked_select(cont_mask).sum() / denom).detach()
        adv = adv - mean_adv

    # 4) Policy gradient style weighted NLL on student
    loss_terms = -adv * stud_lp  # [B, T-1]
    loss = loss_terms.masked_select(cont_mask).mean()

    accelerator.backward(loss)

    # Metrics
    rev_kl = delta.masked_select(cont_mask).mean()
    tokens = cont_mask.sum()
    # Gather metrics across processes
    rev_kl_mean = accelerator.gather_for_metrics(rev_kl.detach()).mean().item()
    tokens_sum = accelerator.gather_for_metrics(tokens.detach()).sum().item()
    loss_val = accelerator.gather_for_metrics(loss.detach()).mean().item()
    return {"loss": float(loss_val), "reverse_kl": float(rev_kl_mean), "tokens": int(tokens_sum)}


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mixed = "bf16" if cfg.dtype.lower() == "bf16" else ("fp16" if cfg.dtype.lower() == "fp16" else "no")
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum, mixed_precision=mixed, log_with=["wandb"] if cfg.wandb_project else None)
    if cfg.wandb_project and accelerator.is_main_process:
        if cfg.wandb_mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = cfg.wandb_mode
        accelerator.init_trackers(cfg.wandb_project, config=vars(cfg), init_kwargs={"wandb": {"name": cfg.wandb_name}})

    # Load models/tokenizer (before prepare for LoRA)
    torch_dtype = (
        torch.bfloat16 if cfg.dtype.lower() == "bf16" else torch.float16 if cfg.dtype.lower() == "fp16" else None
    )
    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_model, torch_dtype=torch_dtype if torch_dtype is not None else None
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model, torch_dtype=torch_dtype if torch_dtype is not None else None
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    try:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            tokenizer.padding_side = "left"
    except Exception:
        pass
    ensure_pad_token(tokenizer)

    # Optional LoRA (apply before prepare)
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未安装，请先 pip install peft")
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        student = get_peft_model(student, lora_cfg)

    # Optimizer
    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Prepare student with Accelerator DDP; keep teacher out of accelerator to avoid DDP replication
    student, optimizer = accelerator.prepare(student, optimizer)

    # Configure teacher for ZeRO-3 inference sharding (no grads, no optimizer)
    for p in teacher.parameters():
        p.requires_grad_(False)
    if cfg.teacher_ds_zero3:
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("需要 deepspeed 用于教师模型 ZeRO-3 分片，请先安装 deepspeed")
        # Default ZeRO-3 inference config if not provided
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {"enabled": cfg.dtype.lower() == "bf16"},
            "fp16": {"enabled": cfg.dtype.lower() == "fp16"},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_gather_16bit_weights_on_model_save": False,
            },
        }
        if cfg.teacher_ds_config and os.path.exists(cfg.teacher_ds_config):
            import json
            with open(cfg.teacher_ds_config, "r") as f:
                ds_cfg = json.load(f)
        teacher, _, _, _ = deepspeed.initialize(model=teacher, model_parameters=None, config=ds_cfg)

    prompts = get_prompts(cfg)
    step = 0
    while step < cfg.steps:
        # Simple round-robin batching over prompts
        start = (step * cfg.batch_size) % max(1, len(prompts))
        end = start + cfg.batch_size
        groups = prompts[start:end] if end <= len(prompts) else (
            prompts[start:] + prompts[: (end % len(prompts))]
        )

        # Shard prompts across processes to avoid duplication
        world = accelerator.num_processes
        rank = accelerator.process_index
        groups_shard = [g for i, g in enumerate(groups) if i % max(1, world) == rank]
        # Expand by group_size to mimic multiple rollouts per prompt
        batch_prompts = [p for p in groups_shard for _ in range(cfg.group_size)]
        # Optional prompt truncation
        if cfg.max_prompt_tokens is not None:
            batch_prompts = [truncate_by_tokens(tokenizer, p, cfg.max_prompt_tokens) for p in batch_prompts]
        with accelerator.accumulate(student):
            optimizer.zero_grad(set_to_none=True)
            metrics = train_step(student, teacher, tokenizer, batch_prompts, cfg, accelerator, optimizer)
            accelerator.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1

        if accelerator.is_main_process and (step % 10 == 0 or step == 1):
            accelerator.print(
                f"step {step:05d} | loss={metrics['loss']:.4f} "
                f"rev_kl={metrics['reverse_kl']:.4f} tokens={metrics['tokens']}"
            )
        if cfg.wandb_project:
            accelerator.log({"train/loss": metrics["loss"], "train/reverse_kl": metrics["reverse_kl"], "train/tokens": metrics["tokens"], "train/step": step}, step=step)

        if accelerator.is_main_process and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            # quick post-KL estimate on a small sample
            with torch.no_grad():
                eval_prompts = prompts[: min(4, len(prompts))]
                seqs, plens = generate_continuations(
                    student, tokenizer, eval_prompts,
                    cfg.max_new_tokens, cfg.temperature, cfg.top_p,
                    cfg.gen_micro_batch,
                    cfg.progress and accelerator.is_main_process,
                )
                am = seqs.ne(tokenizer.pad_token_id).long()
                # micro-batched logprobs for eval as well
                s_parts, t_parts = [], []
                it_eval = range(0, seqs.size(0), max(1, cfg.lp_micro_batch))
                for i in it_eval:
                    sl = slice(i, i + max(1, cfg.lp_micro_batch))
                    s_parts.append(sequence_logprobs(student, seqs[sl], am[sl]))
                    t_parts.append(sequence_logprobs(teacher, seqs[sl], am[sl]))
                stud_lp = torch.cat(s_parts, dim=0)
                teach_lp = torch.cat(t_parts, dim=0)
                # mask continuation
                cont_mask = torch.zeros_like(stud_lp, dtype=torch.bool)
                for i, L in enumerate(plens):
                    start_i = max(L - 1, 0)
                    cont_mask[i, start_i:] = True
                rev_kl_eval = (stud_lp - teach_lp).masked_select(cont_mask).mean().item()
                accelerator.print(f"eval reverse_kl={rev_kl_eval:.4f}")
                if cfg.wandb_project:
                    accelerator.log({"eval/reverse_kl": rev_kl_eval, "train/step": step}, step=step)

        if accelerator.is_main_process and cfg.save_every > 0 and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            # Save adapter if LoRA, else full weights
            to_save = accelerator.unwrap_model(student)
            if cfg.use_lora and PEFT_AVAILABLE and isinstance(to_save, PeftModel):
                to_save.save_pretrained(ckpt_dir)
            else:
                to_save.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            accelerator.print(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    if accelerator.is_main_process:
        # Final save
        to_save = accelerator.unwrap_model(student)
        if cfg.use_lora and PEFT_AVAILABLE and isinstance(to_save, PeftModel):
            to_save.save_pretrained(cfg.output_dir)
        else:
            to_save.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        accelerator.print(f"Training complete. Model saved to {cfg.output_dir}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Local on-policy distillation trainer (no Tinker)")
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./local-distill-out")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--kl_coef", type=float, default=1.0)
    p.add_argument("--kl_discount", type=float, default=0.0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--teacher_ds_zero3", action="store_true", help="使用 DeepSpeed ZeRO-3 对 teacher 做推理分片")
    p.add_argument("--teacher_ds_config", type=str, default=None, help="DeepSpeed 配置文件（可选），未提供则使用内置零三推理配置")
    p.add_argument("--no_progress", action="store_true", help="关闭进度条显示（默认开启，仅主进程显示）")
    p.add_argument("--gen_micro_batch", type=int, default=8, help="生成阶段微批大小")
    p.add_argument("--lp_micro_batch", type=int, default=8, help="logprob 前向微批大小")
    a = p.parse_args()
    return Config(
        student_model=a.student_model,
        teacher_model=a.teacher_model,
        output_dir=a.output_dir,
        steps=a.steps,
        batch_size=a.batch_size,
        group_size=a.group_size,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        kl_coef=a.kl_coef,
        kl_discount=a.kl_discount,
        save_every=a.save_every,
        prompts_file=a.prompts_file,
        seed=a.seed,
        use_lora=a.use_lora,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        dtype=a.dtype,
        grad_accum=a.grad_accum,
        eval_every=a.eval_every,
        dataset=a.dataset,
        max_prompt_tokens=a.max_prompt_tokens,
        wandb_project=a.wandb_project,
        wandb_name=a.wandb_name,
        wandb_mode=a.wandb_mode,
        teacher_ds_zero3=a.teacher_ds_zero3,
        teacher_ds_config=a.teacher_ds_config,
        gen_micro_batch=a.gen_micro_batch,
        lp_micro_batch=a.lp_micro_batch,
        progress=not a.no_progress,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
