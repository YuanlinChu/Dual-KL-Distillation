from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

try:
    from tqdm.auto import tqdm  # type: ignore

    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import deepspeed  # type: ignore

    DEEPSPEED_AVAILABLE = True
except Exception:
    DEEPSPEED_AVAILABLE = False

try:
    import swanlab  # type: ignore

    SWANLAB_AVAILABLE = True
except Exception:
    SWANLAB_AVAILABLE = False

try:
    import yaml  # type: ignore

    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


@dataclass
class Config:
    student_model: str
    teacher_model: str
    output_dir: str
    steps: int = 1000
    batch_size: int = 256
    group_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 1
    top_p: float = 0.95
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    lr_decay: str = "linear"
    min_lr_ratio: float = 0.1
    save_every: int = 100
    prompts_file: str | None = None
    dataset: str | None = None
    dataset_field: str = "question"
    max_prompt_tokens: int | None = None
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    dtype: str = "bf16"
    eval_every: int = 50
    eval_exact_kl: bool = True
    teacher_ds_zero3: bool = False
    teacher_ds_config: str | None = None
    gen_micro_batch: int = 4
    lp_micro_batch: int = 8
    progress: bool = True
    system_prompt: str | None = "Please reason step by step, and put your final answer within \\boxed{{}}."
    print_sample: bool = False
    print_every: int = 10
    debug_mask: bool = False
    loss_fn: str = "ppo"
    num_substeps: int = 1
    ppo_clip_low: float = 0.2
    ppo_clip_high: float = 0.2
    kl_coef: float = 1.0
    kl_discount: float = 0.0
    max_grad_norm: float = 1.0
    lam_r: float = 1.0
    lam_f: float = 0.0
    use_fkl: bool = False
    fkl_decay_until: float = 0.3
    swanlab_project: str | None = None
    swanlab_name: str | None = None
    swanlab_mode: str = "offline"


@dataclass
class RolloutBatch:
    input_ids_cpu: torch.Tensor
    valid_mask_cpu: torch.Tensor
    old_logprobs_cpu: torch.Tensor
    advantages_cpu: torch.Tensor
    teacher_logprobs_cpu: torch.Tensor
    teacher_sampled_tokens_cpu: torch.Tensor
    teacher_sampled_student_logprobs_cpu: torch.Tensor
    teacher_sampled_teacher_logprobs_cpu: torch.Tensor
    prompt_lengths: List[int]
    pad_id: int
    group_count: int
    fkl_weight: float
    sample_prompt: str = ""
    sample_cont: str = ""


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def init_swanlab(cfg: Config) -> None:
    init_kwargs = {
        "project": cfg.swanlab_project,
        "config": vars(cfg),
        "mode": cfg.swanlab_mode,
        "logdir": cfg.output_dir,
    }
    if cfg.swanlab_name:
        for name_key in ("experiment_name", "name", "run_name"):
            try:
                swanlab.init(**init_kwargs, **{name_key: cfg.swanlab_name})
                return
            except TypeError:
                continue
    swanlab.init(**init_kwargs)


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float, decay: str) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = max(0, min(warmup_steps, total_steps))
    step = max(0, min(step, total_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if decay == "none" or total_steps <= warmup_steps:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)
    if decay == "linear":
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    if decay == "cosine":
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return 1.0


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_prompts(path: str | None) -> List[str]:
    if path is None:
        raise ValueError("prompts file path must be provided")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_prompts(cfg: Config) -> List[str]:
    if cfg.prompts_file:
        return load_prompts(cfg.prompts_file)
    if not cfg.dataset:
        raise ValueError("must provide either --prompts_file or a local --dataset path")
    if not os.path.exists(cfg.dataset):
        raise FileNotFoundError(f"dataset path not found: {cfg.dataset}")
    try:
        from datasets import load_from_disk  # type: ignore

        obj = load_from_disk(cfg.dataset)
        if hasattr(obj, "keys"):
            split_name = "train" if "train" in obj.keys() else list(obj.keys())[0]
            ds = obj[split_name]
        else:
            ds = obj
        col = cfg.dataset_field or "question"
        if col in ds.column_names:
            return [str(v) for v in ds[col]]  # type: ignore
        for alt in ["question", "prompt", "input", "text"]:
            if alt in ds.column_names:
                return [str(v) for v in ds[alt]]  # type: ignore
        raise KeyError(f"field `{col}` not found in dataset columns: {list(ds.column_names)}")
    except Exception:
        pass
    try:
        import glob
        import pandas as pd

        parquet_files = sorted(glob.glob(os.path.join(cfg.dataset, "data", "*.parquet")))
        if not parquet_files:
            parquet_files = sorted(glob.glob(os.path.join(cfg.dataset, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"no parquet files found under dataset path: {cfg.dataset}")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        col = cfg.dataset_field or "question"
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
        for alt in ["question", "prompt", "input", "text"]:
            if alt in df.columns:
                return df[alt].dropna().astype(str).tolist()
        raise KeyError(f"field `{col}` not found in parquet columns: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"failed to load prompts from local dataset path `{cfg.dataset}`: {e}") from e


def truncate_by_tokens(tok: PreTrainedTokenizerBase, text: str, max_tokens: int) -> str:
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens])


def apply_chat_format(tok: PreTrainedTokenizerBase, questions: List[str], system_prompt: str | None) -> List[str]:
    if not (hasattr(tok, "apply_chat_template") and callable(getattr(tok, "apply_chat_template"))):
        raise RuntimeError("Tokenizer does not support apply_chat_template; Qwen3 chat template is required.")
    out: List[str] = []
    for q in questions:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": q})
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def per_position_exact_kl(logp_s: torch.Tensor, logp_t: torch.Tensor, kind: str) -> torch.Tensor:
    lps = logp_s[:, :-1, :]
    lpt = logp_t[:, :-1, :]
    ps = lps.exp()
    pt = lpt.exp()
    if kind == "rkl":
        return (ps * (lps - lpt)).sum(dim=-1)
    if kind == "fkl":
        return (pt * (lpt - lps)).sum(dim=-1)
    raise ValueError("kind must be rkl or fkl")


def discounted_future_sum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0:
        return x
    y = torch.zeros_like(x)
    acc = torch.zeros_like(x[..., 0])
    for t in reversed(range(x.shape[-1])):
        acc = x[..., t] + gamma * acc
        y[..., t] = acc
    return y


def scheduled_fkl_weight(step: int, total_steps: int, cfg: Config) -> float:
    if not cfg.use_fkl or cfg.lam_f <= 0.0:
        return 0.0
    if cfg.fkl_decay_until <= 0.0:
        return float(cfg.lam_f)
    progress = float(step) / float(max(1, total_steps))
    if progress >= cfg.fkl_decay_until:
        return 0.0
    remain_ratio = 1.0 - (progress / max(cfg.fkl_decay_until, 1e-8))
    remain_ratio = min(max(remain_ratio, 0.0), 1.0)
    return float(cfg.lam_f) * remain_ratio


def generate_continuations(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    micro_batch: int,
) -> Tuple[torch.Tensor, List[int], int]:
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    all_out_raw: List[torch.Tensor] = []
    all_plen: List[int] = []
    max_t = 0
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    with torch.no_grad():
        for i in range(0, len(prompts), max(1, micro_batch)):
            chunk = prompts[i : i + max(1, micro_batch)]
            batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(device_of(model_for_gen)) for k, v in batch.items()}
            gen = model_for_gen.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )
            max_t = max(max_t, gen.size(1))
            all_out_raw.append(gen.detach())
            all_plen.extend(batch["input_ids"].ne(pad_id).sum(dim=1).tolist())

    def pad_to(t: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
        if t.size(1) == target_len:
            return t
        if t.size(1) > target_len:
            return t[:, :target_len]
        pad_cols = target_len - t.size(1)
        pad = torch.full((t.size(0), pad_cols), pad_token_id, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    if not all_out_raw:
        return torch.empty((0, 0), dtype=torch.long), all_plen, pad_id
    seqs = torch.cat([pad_to(t, max_t, pad_id) for t in all_out_raw], dim=0).cpu()
    return seqs, all_plen, pad_id


def build_continuation_mask(seqs_cpu: torch.Tensor, prompt_lengths: List[int], pad_id: int) -> torch.Tensor:
    bsz, total_t = seqs_cpu.size()
    cont = torch.zeros((bsz, max(total_t - 1, 0)), dtype=torch.bool)
    attn = seqs_cpu.ne(pad_id)
    for i, prompt_len in enumerate(prompt_lengths):
        nonpad = attn[i].nonzero()
        if len(nonpad) == 0:
            continue
        first_nonpad = int(nonpad[0].item())
        start = max(first_nonpad + prompt_len - 1, 0)
        if total_t > 1:
            cont[i, start:] = True
    return cont & attn[:, 1:]


def _gather_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute log P(token_ids) from logits without materializing full log_softmax.

    Uses ``logits.gather() - logsumexp(logits)`` so that only a [B, T, 1]
    intermediate is created instead of [B, T, V].
    """
    gathered = logits[:, :-1, :].gather(-1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits[:, :-1, :], dim=-1)
    return gathered - lse

def gather_action_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return _gather_logprobs(logits, input_ids)

def compute_rollout_logprobs(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    seqs_cpu: torch.Tensor,
    lp_micro_batch: int,
    pad_id: int,
    accelerator: Accelerator,
    compute_fkl_samples: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_chunks: List[torch.Tensor] = []
    teacher_chunks: List[torch.Tensor] = []
    sampled_token_chunks: List[torch.Tensor] = []
    sampled_student_chunks: List[torch.Tensor] = []
    sampled_teacher_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, seqs_cpu.size(0), max(1, lp_micro_batch)):
            sl = slice(i, i + max(1, lp_micro_batch))
            ids_mb = seqs_cpu[sl].to(accelerator.device, non_blocking=True)
            attn_mb = ids_mb.ne(pad_id).long()

            # --- student forward ---
            with accelerator.autocast():
                logits_s = student(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
                student_logprobs = _gather_logprobs(logits_s, ids_mb).float().cpu()
            # keep logits_s only when fkl sampling needs it later
            if not compute_fkl_samples:
                del logits_s

            # --- teacher forward ---
            with accelerator.autocast():
                logits_t = teacher(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
                teacher_logprobs = _gather_logprobs(logits_t, ids_mb).float().cpu()

            old_chunks.append(student_logprobs)
            teacher_chunks.append(teacher_logprobs)

            if compute_fkl_samples:
                # fKL needs full teacher distribution for multinomial sampling
                with accelerator.autocast():
                    logp_t = nn.functional.log_softmax(logits_t[:, :-1, :], dim=-1)
                    probs_t = logp_t.exp()
                    bsz, seq_len, vocab = probs_t.shape
                    sampled = torch.multinomial(probs_t.reshape(-1, vocab), num_samples=1).reshape(bsz, seq_len)
                    del probs_t
                    sampled_token_chunks.append(sampled.cpu())
                    sampled_teacher_chunks.append(
                        logp_t.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).float().cpu()
                    )
                    del logp_t
                    # student: only gather the sampled positions (no full log_softmax)
                    sampled_s_logits = logits_s[:, :-1, :].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                    sampled_s_lse = torch.logsumexp(logits_s[:, :-1, :], dim=-1)
                    sampled_student_chunks.append(
                        (sampled_s_logits - sampled_s_lse).float().cpu()
                    )
                del logits_s

            del ids_mb, attn_mb, logits_t

    old_logprobs = torch.cat(old_chunks, dim=0) if old_chunks else torch.zeros(seqs_cpu.size(0), max(seqs_cpu.size(1) - 1, 0))
    teacher_logprobs = torch.cat(teacher_chunks, dim=0) if teacher_chunks else torch.zeros_like(old_logprobs)

    if compute_fkl_samples and sampled_token_chunks:
        sampled_tokens = torch.cat(sampled_token_chunks, dim=0)
        sampled_student_logprobs = torch.cat(sampled_student_chunks, dim=0)
        sampled_teacher_logprobs = torch.cat(sampled_teacher_chunks, dim=0)
    else:
        seq_len = max(seqs_cpu.size(1) - 1, 0)
        sampled_tokens = torch.zeros(seqs_cpu.size(0), seq_len, dtype=torch.long)
        sampled_student_logprobs = torch.zeros(seqs_cpu.size(0), seq_len)
        sampled_teacher_logprobs = torch.zeros(seqs_cpu.size(0), seq_len)

    return old_logprobs, teacher_logprobs, sampled_tokens, sampled_student_logprobs, sampled_teacher_logprobs            


def build_zero_centered_group_rewards(num_rollouts: int, group_size: int) -> torch.Tensor:
    if group_size <= 1:
        return torch.zeros(num_rollouts, dtype=torch.float32)
    rewards = torch.zeros(num_rollouts, dtype=torch.float32)
    groups = max(1, num_rollouts // group_size)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, num_rollouts)
        rewards[start:end] = rewards[start:end] - rewards[start:end].mean()
    return rewards


def collect_rollout_batch(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    cfg: Config,
    accelerator: Accelerator,
    step: int,
) -> RolloutBatch:
    fkl_weight = scheduled_fkl_weight(step, cfg.steps, cfg)
    seqs_cpu, prompt_lengths, pad_id = generate_continuations(
        student,
        tok,
        prompts,
        cfg.max_new_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.gen_micro_batch,
    )
    valid_mask_cpu = build_continuation_mask(seqs_cpu, prompt_lengths, pad_id)
    (
        old_logprobs_cpu,
        teacher_logprobs_cpu,
        teacher_sampled_tokens_cpu,
        teacher_sampled_student_logprobs_cpu,
        teacher_sampled_teacher_logprobs_cpu,
    ) = compute_rollout_logprobs(
        student,
        teacher,
        seqs_cpu,
        cfg.lp_micro_batch,
        pad_id,
        accelerator,
        compute_fkl_samples=fkl_weight > 0.0,
    )
    advantages_cpu = -cfg.kl_coef * (old_logprobs_cpu - teacher_logprobs_cpu)
    if cfg.kl_discount > 0 and advantages_cpu.numel() > 0:
        discounted_rows: List[torch.Tensor] = []
        for i in range(advantages_cpu.size(0)):
            discounted_rows.append(discounted_future_sum(advantages_cpu[i], cfg.kl_discount))
        advantages_cpu = torch.stack(discounted_rows, dim=0)
    scalar_adv = build_zero_centered_group_rewards(seqs_cpu.size(0), cfg.group_size).unsqueeze(1)
    advantages_cpu = advantages_cpu + scalar_adv
    advantages_cpu = advantages_cpu * valid_mask_cpu.float()

    sample_prompt = ""
    sample_cont = ""
    if seqs_cpu.size(0) > 0:
        ids_0 = seqs_cpu[0]
        nonpad = ids_0.ne(pad_id).nonzero()
        if len(nonpad) > 0:
            first_nonpad = int(nonpad[0].item())
            end = int(nonpad[-1].item() + 1)
            prompt_end = max(first_nonpad + prompt_lengths[0], first_nonpad)
            sample_prompt = tok.decode(ids_0[first_nonpad:prompt_end].tolist())
            if end > prompt_end:
                sample_cont = tok.decode(ids_0[prompt_end:end].tolist())

    if cfg.debug_mask and seqs_cpu.size(0) > 0:
        prompt_ids = tok(prompts[0], return_tensors="pt", truncation=True)["input_ids"][0].cpu()
        nonpad = seqs_cpu[0].ne(pad_id).nonzero()
        if len(nonpad) == 0:
            raise RuntimeError("debug_mask: no non-pad tokens")
        first_nonpad = int(nonpad[0].item())
        got_prompt = seqs_cpu[0, first_nonpad : first_nonpad + prompt_ids.numel()]
        if not torch.equal(got_prompt, prompt_ids):
            raise RuntimeError("debug_mask: prompt ids do not match rollout prefix")

    return RolloutBatch(
        input_ids_cpu=seqs_cpu,
        valid_mask_cpu=valid_mask_cpu,
        old_logprobs_cpu=old_logprobs_cpu,
        advantages_cpu=advantages_cpu,
        teacher_logprobs_cpu=teacher_logprobs_cpu,
        teacher_sampled_tokens_cpu=teacher_sampled_tokens_cpu,
        teacher_sampled_student_logprobs_cpu=teacher_sampled_student_logprobs_cpu,
        teacher_sampled_teacher_logprobs_cpu=teacher_sampled_teacher_logprobs_cpu,
        prompt_lengths=prompt_lengths,
        pad_id=pad_id,
        group_count=max(1, seqs_cpu.size(0) // max(1, cfg.group_size)),
        fkl_weight=fkl_weight,
        sample_prompt=sample_prompt,
        sample_cont=sample_cont,
    )


def rl_update_substep(
    student: PreTrainedModel,
    rollout: RolloutBatch,
    cfg: Config,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
) -> dict:
    student.train()
    total_valid_tokens = int(rollout.valid_mask_cpu.sum().item())
    if total_valid_tokens == 0:
        return {
            "loss": 0.0,
            "ratio": 1.0,
            "clip_frac": 0.0,
            "approx_kl": 0.0,
            "entropy": 0.0,
            "fkl_loss": 0.0,
            "tokens": 0,
            "grad_norm": 0.0,
        }

    loss_sum = torch.tensor(0.0, device=accelerator.device)
    ratio_sum = torch.tensor(0.0, device=accelerator.device)
    clip_sum = torch.tensor(0.0, device=accelerator.device)
    approx_kl_sum = torch.tensor(0.0, device=accelerator.device)
    entropy_sum = torch.tensor(0.0, device=accelerator.device)
    fkl_loss_sum = torch.tensor(0.0, device=accelerator.device)
    token_sum = torch.tensor(0.0, device=accelerator.device)
    use_fkl = rollout.fkl_weight > 0.0

    mb = max(1, cfg.lp_micro_batch)
    optimizer.zero_grad(set_to_none=True)
    for i in range(0, rollout.input_ids_cpu.size(0), mb):
        sl = slice(i, i + mb)
        ids_mb = rollout.input_ids_cpu[sl].to(accelerator.device, non_blocking=True)
        attn_mb = ids_mb.ne(rollout.pad_id).long()
        valid_mb = rollout.valid_mask_cpu[sl].to(accelerator.device)
        old_lp_mb = rollout.old_logprobs_cpu[sl].to(accelerator.device)
        adv_mb = rollout.advantages_cpu[sl].to(accelerator.device)
        if use_fkl:
            teacher_tokens_mb = rollout.teacher_sampled_tokens_cpu[sl].to(accelerator.device)
        else:
            teacher_tokens_mb = None

        with accelerator.autocast():
            logits_s = student(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
            logits_shift = logits_s[:, :-1, :]
            lse = torch.logsumexp(logits_shift, dim=-1)

            # PPO ratio: only gather the action token logprobs, no full log_softmax
            cur_lp_mb = logits_shift.gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1) - lse
            ratio = torch.exp(cur_lp_mb - old_lp_mb)
            if cfg.loss_fn == "ppo":
                clip_low = 1.0 - cfg.ppo_clip_low
                clip_high = 1.0 + cfg.ppo_clip_high
                clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
                obj = torch.min(ratio * adv_mb, clipped_ratio * adv_mb)
                loss_pos = -obj
                clip_frac = ((ratio < clip_low) | (ratio > clip_high)).float()
            elif cfg.loss_fn == "importance_sampling":
                loss_pos = -(ratio * adv_mb)
                clip_frac = torch.zeros_like(ratio)
            else:
                raise ValueError(f"unsupported loss_fn: {cfg.loss_fn}")

            # Entropy: compute without materializing full probs tensor
            # H = logsumexp - (sum(p * logit) / sum(p)) = logsumexp - E[logit]
            # = logsumexp(logits) - sum(softmax(logits) * logits)
            # Use: H = log(sum(exp(logits))) - sum(exp(logits - lse) * logits) / 1
            with torch.no_grad():
                softmax_shift = torch.softmax(logits_shift, dim=-1)
                entropy_pos = lse - (softmax_shift * logits_shift).sum(dim=-1)
                del softmax_shift

            total_loss_pos = cfg.lam_r * loss_pos
            fkl_loss_pos = torch.zeros_like(total_loss_pos)
            if use_fkl and teacher_tokens_mb is not None:
                cur_teacher_lp = logits_shift.gather(-1, teacher_tokens_mb.unsqueeze(-1)).squeeze(-1) - lse
                fkl_loss_pos = -cur_teacher_lp
                total_loss_pos = total_loss_pos + rollout.fkl_weight * fkl_loss_pos
            del logits_shift, lse
            loss_mb = total_loss_pos.masked_select(valid_mb).sum() / float(max(1, total_valid_tokens))

        is_last = (i + mb) >= rollout.input_ids_cpu.size(0)
        if not is_last:
            with accelerator.no_sync(student):
                accelerator.backward(loss_mb)
        else:
            accelerator.backward(loss_mb)

        ratio_sum = ratio_sum + ratio.masked_select(valid_mb).detach().sum()
        clip_sum = clip_sum + clip_frac.masked_select(valid_mb).detach().sum()
        approx_kl_sum = approx_kl_sum + (old_lp_mb - cur_lp_mb).masked_select(valid_mb).detach().sum()
        entropy_sum = entropy_sum + entropy_pos.masked_select(valid_mb).detach().sum()
        if use_fkl:
            fkl_loss_sum = fkl_loss_sum + fkl_loss_pos.masked_select(valid_mb).detach().sum()
        token_sum = token_sum + valid_mb.sum().detach()
        loss_sum = loss_sum + loss_mb.detach()

        del ids_mb, attn_mb, valid_mb, old_lp_mb, adv_mb, logits_s, cur_lp_mb, ratio, loss_pos, clip_frac, entropy_pos, loss_mb, total_loss_pos, fkl_loss_pos, teacher_tokens_mb

    grad_norm = accelerator.clip_grad_norm_(student.parameters(), max_norm=cfg.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    gathered_tokens = accelerator.gather_for_metrics(token_sum).sum().clamp_min(1)
    return {
        "loss": float(accelerator.gather_for_metrics(loss_sum).mean().item()),
        "ratio": float((accelerator.gather_for_metrics(ratio_sum).sum() / gathered_tokens).item()),
        "clip_frac": float((accelerator.gather_for_metrics(clip_sum).sum() / gathered_tokens).item()),
        "approx_kl": float((accelerator.gather_for_metrics(approx_kl_sum).sum() / gathered_tokens).item()),
        "entropy": float((accelerator.gather_for_metrics(entropy_sum).sum() / gathered_tokens).item()),
        "fkl_loss": float((accelerator.gather_for_metrics(fkl_loss_sum).sum() / gathered_tokens).item()),
        "tokens": int(accelerator.gather_for_metrics(token_sum).sum().item()),
        "grad_norm": float(grad_norm),
    }


def rollout_metrics(rollout: RolloutBatch, accelerator: Accelerator) -> dict:
    valid = rollout.valid_mask_cpu
    tokens = int(valid.sum().item())
    if tokens == 0:
        return {
            "reverse_kl": 0.0,
            "forward_kl_mc": 0.0,
            "fkl_weight": rollout.fkl_weight,
            "reward": 0.0,
            "advantages": 0.0,
            "tokens": 0,
        }
    rkl = ((rollout.old_logprobs_cpu - rollout.teacher_logprobs_cpu) * valid.float()).sum() / max(1, tokens)
    if rollout.fkl_weight > 0.0:
        fkl = (
            (rollout.teacher_sampled_teacher_logprobs_cpu - rollout.teacher_sampled_student_logprobs_cpu) * valid.float()
        ).sum() / max(1, tokens)
        fkl_val = float(fkl.item())
    else:
        fkl_val = 0.0
    adv = rollout.advantages_cpu.masked_select(valid).mean()
    return {
        "reverse_kl": float(rkl.item()),
        "forward_kl_mc": fkl_val,
        "fkl_weight": rollout.fkl_weight,
        "reward": 0.0,
        "advantages": float(adv.item()),
        "tokens": tokens,
    }


def evaluate_exact_kl(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    cfg: Config,
    accelerator: Accelerator,
) -> tuple[float, float]:
    if not prompts:
        return 0.0, 0.0
    with torch.no_grad():
        seqs_cpu, prompt_lengths, pad_id = generate_continuations(
            student, tok, prompts, cfg.max_new_tokens, cfg.temperature, cfg.top_p, cfg.gen_micro_batch
        )
        valid_mask = build_continuation_mask(seqs_cpu, prompt_lengths, pad_id)
        r_sum = torch.tensor(0.0, device=accelerator.device)
        f_sum = torch.tensor(0.0, device=accelerator.device)
        t_sum = torch.tensor(0.0, device=accelerator.device)
        for i in range(0, seqs_cpu.size(0), max(1, cfg.lp_micro_batch)):
            sl = slice(i, i + max(1, cfg.lp_micro_batch))
            ids_mb = seqs_cpu[sl].to(accelerator.device, non_blocking=True)
            am_mb = ids_mb.ne(pad_id).long()
            valid_mb = valid_mask[sl].to(accelerator.device)
            with accelerator.autocast():
                logits_s = student(input_ids=ids_mb, attention_mask=am_mb, use_cache=False).logits
                logp_s = nn.functional.log_softmax(logits_s, dim=-1)
                del logits_s
                logits_t = teacher(input_ids=ids_mb, attention_mask=am_mb, use_cache=False).logits
                logp_t = nn.functional.log_softmax(logits_t, dim=-1)
                del logits_t
            r_pos = per_position_exact_kl(logp_s, logp_t, kind="rkl")
            f_pos = per_position_exact_kl(logp_s, logp_t, kind="fkl")
            del logp_s, logp_t
            r_sum = r_sum + r_pos.masked_select(valid_mb).sum()
            f_sum = f_sum + f_pos.masked_select(valid_mb).sum()
            t_sum = t_sum + valid_mb.sum()
        r_all = accelerator.gather_for_metrics(r_sum).sum()
        f_all = accelerator.gather_for_metrics(f_sum).sum()
        t_all = accelerator.gather_for_metrics(t_sum).sum()
        rkl_exact = (r_all / t_all).item() if t_all.item() > 0 else 0.0
        fkl_exact = (f_all / t_all).item() if t_all.item() > 0 else 0.0
        return rkl_exact, fkl_exact


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mixed = "bf16" if cfg.dtype.lower() == "bf16" else ("fp16" if cfg.dtype.lower() == "fp16" else "no")
    accelerator = Accelerator(mixed_precision=mixed)
    if cfg.swanlab_project and accelerator.is_main_process:
        if not SWANLAB_AVAILABLE:
            raise RuntimeError("swanlab 未安装，请先 pip install swanlab")
        if cfg.swanlab_mode in ("offline", "disabled"):
            os.environ["SWANLAB_MODE"] = cfg.swanlab_mode
        init_swanlab(cfg)

    torch_dtype = torch.bfloat16 if cfg.dtype.lower() == "bf16" else torch.float16 if cfg.dtype.lower() == "fp16" else None
    student = AutoModelForCausalLM.from_pretrained(cfg.student_model, torch_dtype=torch_dtype if torch_dtype else None)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, torch_dtype=torch_dtype if torch_dtype else None)
    tok = AutoTokenizer.from_pretrained(cfg.student_model)
    ensure_pad_token(tok)
    try:
        tok.padding_side = "left"
    except Exception:
        pass

    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未安装，请先 pip install peft")
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        student = get_peft_model(student, lora_cfg)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay)
    student, optimizer = accelerator.prepare(student, optimizer)

    for p in teacher.parameters():
        p.requires_grad_(False)
    if cfg.teacher_ds_zero3:
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("需要 deepspeed 用于教师模型 ZeRO-3 分片，请先安装 deepspeed")
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
        teacher, _, _, _ = deepspeed.initialize(model=teacher, model_parameters=None, config=ds_cfg)

    prompts = get_prompts(cfg)
    total_optim_steps = max(1, cfg.steps * cfg.num_substeps)
    warmup_steps = cfg.warmup_steps if cfg.warmup_steps > 0 else int(cfg.warmup_ratio * total_optim_steps)
    warmup_steps = max(0, min(warmup_steps, total_optim_steps))

    global_bar = None
    if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
        global_bar = tqdm(total=cfg.steps, desc="train", dynamic_ncols=True)

    optim_step = 0
    for step in range(1, cfg.steps + 1):
        start = ((step - 1) * cfg.batch_size) % max(1, len(prompts))
        end = start + cfg.batch_size
        groups = prompts[start:end] if end <= len(prompts) else (prompts[start:] + prompts[: (end % len(prompts))])
        world = accelerator.num_processes
        rank = accelerator.process_index
        groups_shard = [g for i, g in enumerate(groups) if i % max(1, world) == rank]
        batch_prompts = [p for p in groups_shard for _ in range(cfg.group_size)]
        batch_prompts = apply_chat_format(tok, batch_prompts, cfg.system_prompt)
        if cfg.max_prompt_tokens is not None:
            batch_prompts = [truncate_by_tokens(tok, p, cfg.max_prompt_tokens) for p in batch_prompts]

        rollout = collect_rollout_batch(student, teacher, tok, batch_prompts, cfg, accelerator, step)
        rollout_stat = rollout_metrics(rollout, accelerator)

        substep_metrics: List[dict] = []
        for _ in range(cfg.num_substeps):
            optim_step += 1
            lr_mult = lr_multiplier(optim_step, total_optim_steps, warmup_steps, cfg.min_lr_ratio, cfg.lr_decay)
            current_lr = cfg.learning_rate * lr_mult
            set_optimizer_lr(optimizer, current_lr)
            substep_metrics.append(rl_update_substep(student, rollout, cfg, accelerator, optimizer))

        mean_loss = sum(m["loss"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_ratio = sum(m["ratio"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_clip = sum(m["clip_frac"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_post_kl = sum(m["approx_kl"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_entropy = sum(m["entropy"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_grad = sum(m["grad_norm"] for m in substep_metrics) / max(1, len(substep_metrics))
        mean_fkl_loss = sum(m["fkl_loss"] for m in substep_metrics) / max(1, len(substep_metrics))

        if accelerator.is_main_process and cfg.print_sample and step % cfg.print_every == 0:
            if rollout.sample_prompt:
                accelerator.print("[sample_prompt]")
                accelerator.print(rollout.sample_prompt)
            if rollout.sample_cont:
                accelerator.print("[sample_cont]")
                accelerator.print(rollout.sample_cont)

        if accelerator.is_main_process and (step % 10 == 0 or step == 1):
            msg = (
                f"step {step:05d}/{cfg.steps:05d} | loss={mean_loss:.4f} "
                f"rkl={rollout_stat['reverse_kl']:.4f} ratio={mean_ratio:.4f} "
                f"fkl={rollout_stat['forward_kl_mc']:.4f} fkl_w={rollout_stat['fkl_weight']:.4f} "
                f"fkl_loss={mean_fkl_loss:.4f} clip={mean_clip:.4f} "
                f"post_kl={mean_post_kl:.4f} tok={rollout_stat['tokens']}"
            )
            if global_bar is not None:
                global_bar.set_postfix_str(
                    f"loss={mean_loss:.4f} rkl={rollout_stat['reverse_kl']:.4f} fkl_w={rollout_stat['fkl_weight']:.3f} lr={current_lr:.2e}"
                )
            else:
                accelerator.print(msg)
        if cfg.swanlab_project and accelerator.is_main_process:
            swanlab.log(
                {
                    "train/loss": mean_loss,
                    "train/reverse_kl": rollout_stat["reverse_kl"],
                    "train/forward_kl_mc": rollout_stat["forward_kl_mc"],
                    "train/fkl_loss": mean_fkl_loss,
                    "train/fkl_weight": rollout_stat["fkl_weight"],
                    "train/ratio": mean_ratio,
                    "train/clip_frac": mean_clip,
                    "train/approx_kl": mean_post_kl,
                    "train/entropy": mean_entropy,
                    "train/grad_norm": mean_grad,
                    "train/lr": float(current_lr),
                    "train/advantages": rollout_stat["advantages"],
                    "train/tokens": rollout_stat["tokens"],
                    "train/step": step,
                },
                step=step,
            )

        if global_bar is not None:
            global_bar.update(1)

        if cfg.eval_exact_kl and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            eval_prompts = prompts[-min(4, len(prompts)) :]
            eval_prompts = apply_chat_format(tok, eval_prompts, cfg.system_prompt) if eval_prompts else []
            rkl_exact, fkl_exact = evaluate_exact_kl(student, teacher, tok, eval_prompts, cfg, accelerator)
            if accelerator.is_main_process:
                accelerator.print(f"eval rkl_exact={rkl_exact:.4f} fkl_exact={fkl_exact:.4f}")
            if cfg.swanlab_project and accelerator.is_main_process:
                swanlab.log(
                    {
                        "eval/rkl_exact": rkl_exact,
                        "eval/fkl_exact": fkl_exact,
                        "train/step": step,
                    },
                    step=step,
                )

        if accelerator.is_main_process and cfg.save_every > 0 and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = accelerator.unwrap_model(student)
            if cfg.use_lora and PEFT_AVAILABLE and isinstance(to_save, PeftModel):
                to_save.save_pretrained(ckpt_dir)
            else:
                to_save.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            accelerator.print(f"Saved checkpoint to {ckpt_dir}")

    if accelerator.is_main_process:
        if global_bar is not None:
            global_bar.close()
        to_save = accelerator.unwrap_model(student)
        if cfg.use_lora and PEFT_AVAILABLE and isinstance(to_save, PeftModel):
            to_save.save_pretrained(cfg.output_dir)
        else:
            to_save.save_pretrained(cfg.output_dir)
        tok.save_pretrained(cfg.output_dir)
        accelerator.print(f"Training complete. Model saved to {cfg.output_dir}")
        if cfg.swanlab_project and SWANLAB_AVAILABLE:
            try:
                finish = getattr(swanlab, "finish", None)
                if callable(finish):
                    finish()
            except Exception as e:
                accelerator.print(f"[warn] swanlab.finish() failed: {e}")


def _load_yaml_config(path: str, experiment: str | None) -> Dict[str, Any]:
    if not YAML_AVAILABLE:
        raise RuntimeError("需要 PyYAML 才能加载配置文件，请先安装 `pyyaml`。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping at top level")
    base = raw.get("base", {})
    if base is None:
        base = {}
    if not isinstance(base, dict):
        raise ValueError("`base` in YAML config must be a mapping")
    merged = dict(base)
    if experiment:
        exp_cfg = raw.get(experiment)
        if exp_cfg is None:
            available = [k for k, v in raw.items() if k != "base" and isinstance(v, dict)]
            raise KeyError(f"experiment `{experiment}` not found in YAML. Available: {available}")
        if not isinstance(exp_cfg, dict):
            raise ValueError(f"experiment `{experiment}` must map to a config object")
        merged.update(exp_cfg)
    return merged


def parse_args() -> Config:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    pre.add_argument("--experiment", type=str, default=None, help="YAML 中的实验名；不传则只使用 base")
    pre_args, remaining = pre.parse_known_args()
    yaml_defaults: Dict[str, Any] = {}
    if pre_args.config:
        yaml_defaults = _load_yaml_config(pre_args.config, pre_args.experiment)

    p = argparse.ArgumentParser(
        description="RL-style local on-policy distillation trainer",
        parents=[pre],
    )
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./opd-rl-out")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_decay", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--dataset_field", type=str, default="question")
    p.add_argument("--max_prompt_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--no_eval_exact_kl", action="store_true")
    p.add_argument("--teacher_ds_zero3", action="store_true")
    p.add_argument("--teacher_ds_config", type=str, default=None)
    p.add_argument("--gen_micro_batch", type=int, default=4)
    p.add_argument("--lp_micro_batch", type=int, default=8)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--system_prompt", type=str, default="Please reason step by step, and put your final answer within \\boxed{{}}.")
    p.add_argument("--print_sample", action="store_true")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--debug_mask", action="store_true")
    p.add_argument("--loss_fn", type=str, default="ppo", choices=["ppo", "importance_sampling"])
    p.add_argument("--num_substeps", type=int, default=1)
    p.add_argument("--ppo_clip_low", type=float, default=0.2)
    p.add_argument("--ppo_clip_high", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=1.0)
    p.add_argument("--kl_discount", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_f", type=float, default=0.0)
    p.add_argument("--use_fkl", action="store_true")
    p.add_argument("--fkl_decay_until", type=float, default=0.3, help="fKL 线性衰减到 0 的训练进度比例；0 表示不衰减")
    p.add_argument("--swanlab_project", type=str, default=None)
    p.add_argument("--swanlab_name", type=str, default=None)
    p.add_argument("--swanlab_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    if yaml_defaults:
        p.set_defaults(**yaml_defaults)
        for action in p._actions:
            if action.dest in ("student_model", "teacher_model"):
                if action.dest in yaml_defaults:
                    action.required = False
    a = p.parse_args(remaining)
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
        warmup_steps=a.warmup_steps,
        warmup_ratio=a.warmup_ratio,
        lr_decay=a.lr_decay,
        min_lr_ratio=a.min_lr_ratio,
        save_every=a.save_every,
        prompts_file=a.prompts_file,
        dataset=a.dataset,
        dataset_field=a.dataset_field,
        max_prompt_tokens=a.max_prompt_tokens,
        seed=a.seed,
        use_lora=bool(a.use_lora),
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        dtype=a.dtype,
        eval_every=a.eval_every,
        eval_exact_kl=not a.no_eval_exact_kl,
        teacher_ds_zero3=bool(a.teacher_ds_zero3),
        teacher_ds_config=a.teacher_ds_config,
        gen_micro_batch=a.gen_micro_batch,
        lp_micro_batch=a.lp_micro_batch,
        progress=not a.no_progress,
        system_prompt=a.system_prompt,
        print_sample=bool(a.print_sample),
        print_every=a.print_every,
        debug_mask=bool(a.debug_mask),
        loss_fn=a.loss_fn,
        num_substeps=max(1, a.num_substeps),
        ppo_clip_low=max(0.0, a.ppo_clip_low),
        ppo_clip_high=max(0.0, a.ppo_clip_high),
        kl_coef=a.kl_coef,
        kl_discount=a.kl_discount,
        max_grad_norm=a.max_grad_norm,
        lam_r=max(0.0, a.lam_r),
        lam_f=max(0.0, a.lam_f),
        use_fkl=bool(a.use_fkl),
        fkl_decay_until=max(0.0, a.fkl_decay_until),
        swanlab_project=a.swanlab_project,
        swanlab_name=a.swanlab_name,
        swanlab_mode=a.swanlab_mode,
    )


if __name__ == "__main__":
    main(parse_args())
