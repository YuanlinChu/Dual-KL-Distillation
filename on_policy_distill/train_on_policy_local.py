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
    dataset_field: str = "question"  # 当 dataset 为本地 HF 数据集目录时，抽取文本字段
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
    # 若 --dataset 传入的是一个本地目录，则尝试从磁盘加载 HF 数据集
    if cfg.dataset and os.path.exists(cfg.dataset):
        try:
            from datasets import load_from_disk  # type: ignore

            obj = load_from_disk(cfg.dataset)
            # DatasetDict 或 Dataset
            if hasattr(obj, "keys"):
                split_name = "train" if "train" in obj.keys() else list(obj.keys())[0]
                ds = obj[split_name]
            else:
                ds = obj
            field = cfg.dataset_field or "question"
            # 优先取指定字段，其次尝试常见字段
            if field in ds.column_names:
                return [str(v) for v in ds[field]]  # type: ignore
            for alt in ["question", "prompt", "input", "text"]:
                if alt in ds.column_names:
                    return [str(v) for v in ds[alt]]  # type: ignore
        except Exception:
            pass
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


def per_position_exact_kl(logp_s: torch.Tensor, logp_t: torch.Tensor, kind: str) -> torch.Tensor:
    """逐位置精确 KL（不聚合），返回 [B, T-1]

    kind="rkl": KL(p_s || p_t) = sum_v p_s(v)*(log p_s - log p_t)
    kind="fkl": KL(p_t || p_s) = sum_v p_t(v)*(log p_t - log p_s)
    与 next-token 对齐：去掉最后一位预测。
    """
    lps = logp_s[:, :-1, :]
    lpt = logp_t[:, :-1, :]
    ps = lps.exp()
    pt = lpt.exp()
    if kind == "rkl":
        return (ps * (lps - lpt)).sum(dim=-1)
    elif kind == "fkl":
        return (pt * (lpt - lps)).sum(dim=-1)
    else:
        raise ValueError("kind must be rkl or fkl")


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
) -> Tuple[torch.Tensor, List[int], int]:
    # unwrap if DDP-wrapped
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    outs: List[torch.Tensor] = []
    plens: List[int] = []
    max_seq_len: int = 0
    pad_id: int = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    with torch.no_grad():
        iterator = range(0, len(prompts), max(1, micro_batch))
        # 移除微批次级别进度条（统一由全局训练进度条展示）
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
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outs.append(gen.detach().cpu())
            # Track the maximum generated sequence length across micro-batches
            if gen.size(1) > max_seq_len:
                max_seq_len = gen.size(1)
            plens.extend(batch["input_ids"].ne(pad_id).sum(dim=1).tolist())
        # 无微批进度条更新/关闭
    # Pad all micro-batch outputs to the same length before concatenation (CPU tensors)
    if len(outs) == 0:
        return torch.empty(0, dtype=torch.long), plens, pad_id
    if max_seq_len > 0:
        for j in range(len(outs)):
            seq = outs[j]
            cur_len = seq.size(1)
            if cur_len < max_seq_len:
                padded = torch.full(
                    (seq.size(0), max_seq_len),
                    fill_value=pad_id,
                    dtype=seq.dtype,
                    device=seq.device,
                )
                padded[:, :cur_len] = seq
                outs[j] = padded
    return torch.cat(outs, dim=0), plens, pad_id


def sequence_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    use_cache: bool | None = None,
) -> torch.Tensor:
    """Return per-token logprobs for targets (shifted) with shape [B, T-1]."""
    kwargs = {}
    if use_cache is not None:
        kwargs["use_cache"] = use_cache
    logits = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
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
    full_seqs_cpu, prompt_lengths, pad_id = generate_continuations(
        student, tokenizer, prompts,
        cfg.max_new_tokens, cfg.temperature, cfg.top_p,
        cfg.gen_micro_batch,
        cfg.progress and accelerator.is_main_process,
    )

    # 注意：生成结果驻留在 CPU；后续前向时分片搬到 GPU

    # 2) 构造续写掩码与总有效 token 数（续写且非 pad）
    B = full_seqs_cpu.size(0)
    T = full_seqs_cpu.size(1)
    # cont_mask 形状 [B, T-1]
    cont_mask = torch.zeros((B, T - 1), dtype=torch.bool)
    for i, L in enumerate(prompt_lengths):
        start = max(L - 1, 0)
        if start < T - 1:
            cont_mask[i, start:] = True
    attn_full = full_seqs_cpu.ne(pad_id)
    valid_mask_full = cont_mask & attn_full[:, 1:].bool()
    tokens_total = int(valid_mask_full.sum().item())
    if tokens_total == 0:
        return {"loss": 0.0, "reverse_kl": 0.0, "tokens": 0}

    # 3) 微批前向 + 反向（与 dual_kl 微批反向对齐）
    teacher.eval()
    student.train()
    mb = max(1, cfg.lp_micro_batch)

    # 指标累计
    d_rkl_sum = torch.tensor(0.0, device=accelerator.device)
    tokens_accum = torch.tensor(0.0, device=accelerator.device)
    loss_sum = torch.tensor(0.0, device=accelerator.device)

    for i in range(0, B, mb):
        sl = slice(i, i + mb)
        ids_mb = full_seqs_cpu[sl].to(accelerator.device, non_blocking=True)
        attn_mb = ids_mb.ne(pad_id).long()
        valid_mask_mb = (cont_mask[sl].to(accelerator.device)) & attn_mb[:, 1:].bool()

        with accelerator.autocast():
            with torch.no_grad():
                logits_t = teacher(input_ids=ids_mb, attention_mask=attn_mb, use_cache=True).logits
                logp_t_mb = nn.functional.log_softmax(logits_t, dim=-1)
            logits_s = student(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
            logp_s_mb = nn.functional.log_softmax(logits_s, dim=-1)
        del logits_t, logits_s

        # 反向 KL 的 MC 优势（detach），可选时序折扣
        s_g_mb = logp_s_mb[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        t_g_mb = logp_t_mb[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        d_rkl_mb = (s_g_mb.detach() - t_g_mb)  # 仅作指标
        adv_mb = -(s_g_mb - t_g_mb).detach() * cfg.kl_coef
        if cfg.kl_discount > 0:
            adv_mb = discounted_future_sum(adv_mb, cfg.kl_discount)
        rkl_loss_pos = -adv_mb * s_g_mb

        # 按整批 token 总数归一化，逐片反向；除最后一片外 no_sync 降低 AllReduce
        loss_mb = rkl_loss_pos.masked_select(valid_mask_mb).sum() / float(tokens_total)
        is_last = (i + mb) >= B
        if not is_last:
            with accelerator.no_sync(student):
                accelerator.backward(loss_mb)
        else:
            accelerator.backward(loss_mb)

        # 累计指标（标量），用于最终聚合
        d_rkl_sum = d_rkl_sum + d_rkl_mb.masked_select(valid_mask_mb).detach().sum()
        tokens_accum = tokens_accum + valid_mask_mb.sum().detach()
        loss_sum = loss_sum + loss_mb.detach()

        # 移除微批次进度条更新

        # 释放切片张量
        del ids_mb, attn_mb, valid_mask_mb, logp_t_mb, logp_s_mb, s_g_mb, t_g_mb, rkl_loss_pos, loss_mb, d_rkl_mb, adv_mb

    # 无微批次进度条关闭

    # 跨进程聚合指标
    rev_kl_mean = (
        accelerator.gather_for_metrics(d_rkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum()
    ).item()
    tokens = int(accelerator.gather_for_metrics(tokens_accum).sum().item())
    loss_val = accelerator.gather_for_metrics(loss_sum).mean().item()
    return {"loss": float(loss_val), "reverse_kl": float(rev_kl_mean), "tokens": tokens}


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
        cfg.student_model, dtype=torch_dtype if torch_dtype is not None else None
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model, dtype=torch_dtype if torch_dtype is not None else None
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
    # 全局训练进度条（当前 step/总 step）
    global_bar = None
    if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
        global_bar = tqdm(total=cfg.steps, desc="train", dynamic_ncols=True)
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
        if global_bar is not None:
            global_bar.update(1)

        if accelerator.is_main_process and (step % 10 == 0 or step == 1):
            msg = (
                f"step {step:05d}/{cfg.steps:05d} | "
                f"loss={metrics['loss']:.4f} rev_kl={metrics['reverse_kl']:.4f} tokens={metrics['tokens']}"
            )
            if global_bar is not None:
                global_bar.set_postfix_str(
                    f"loss={metrics['loss']:.4f} rkl={metrics['reverse_kl']:.4f} tok={metrics['tokens']}"
                )
            else:
                accelerator.print(msg)
        if cfg.wandb_project:
            accelerator.log({"train/loss": metrics["loss"], "train/reverse_kl": metrics["reverse_kl"], "train/tokens": metrics["tokens"], "train/step": step}, step=step)

        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            # 在所有进程上执行 exact KL eval；只在主进程打印
            with torch.no_grad():
                k = min(4, len(prompts))
                eval_prompts = prompts[-k:] if k > 0 else []
                if eval_prompts:
                    seqs_cpu, plens, pad_id = generate_continuations(
                        student, tokenizer, eval_prompts,
                        cfg.max_new_tokens, cfg.temperature, cfg.top_p,
                        cfg.gen_micro_batch,
                        cfg.progress and accelerator.is_main_process,
                    )
                    r_sum = torch.tensor(0.0, device=accelerator.device)
                    f_sum = torch.tensor(0.0, device=accelerator.device)
                    t_sum = torch.tensor(0.0, device=accelerator.device)
                    it_eval = range(0, seqs_cpu.size(0), max(1, cfg.lp_micro_batch))
                    for i_eval in it_eval:
                        sl = slice(i_eval, i_eval + max(1, cfg.lp_micro_batch))
                        ids_mb = seqs_cpu[sl].to(accelerator.device, non_blocking=True)
                        am_mb = ids_mb.ne(pad_id).long()
                        with accelerator.autocast():
                            logits_s = student(input_ids=ids_mb, attention_mask=am_mb, use_cache=False).logits
                            logits_t = teacher(input_ids=ids_mb, attention_mask=am_mb, use_cache=True).logits
                            logp_s = nn.functional.log_softmax(logits_s, dim=-1)
                            logp_t = nn.functional.log_softmax(logits_t, dim=-1)
                        # exact per-position KL
                        r_pos = per_position_exact_kl(logp_s, logp_t, kind="rkl")
                        f_pos = per_position_exact_kl(logp_s, logp_t, kind="fkl")
                        cont_mask = torch.zeros_like(r_pos, dtype=torch.bool)
                        for j, L in enumerate(plens[sl]):
                            start_j = max(L - 1, 0)
                            cont_mask[j, start_j:] = True
                        valid_mask = cont_mask & am_mb[:, 1:].bool()
                        r_sum = r_sum + r_pos.masked_select(valid_mask).sum()
                        f_sum = f_sum + f_pos.masked_select(valid_mask).sum()
                        t_sum = t_sum + valid_mask.sum()
                    r_all = accelerator.gather_for_metrics(r_sum).sum()
                    f_all = accelerator.gather_for_metrics(f_sum).sum()
                    t_all = accelerator.gather_for_metrics(t_sum).sum()
                    rkl_exact = (r_all / t_all).item() if t_all.item() > 0 else 0.0
                    fkl_exact = (f_all / t_all).item() if t_all.item() > 0 else 0.0
                    if accelerator.is_main_process:
                        accelerator.print(f"eval rkl_exact={rkl_exact:.4f} fkl_exact={fkl_exact:.4f}")
                    if cfg.wandb_project:
                        accelerator.log({"eval/rkl_exact": rkl_exact, "eval/fkl_exact": fkl_exact, "train/step": step}, step=step)

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
        if global_bar is not None:
            global_bar.close()
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
    p.add_argument(
        "--dataset_field",
        type=str,
        default="question",
        help="当 --dataset 指向本地 HF 数据集目录时，使用该列名作为用户提示字段，默认 question",
    )
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
        dataset_field=a.dataset_field,
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
