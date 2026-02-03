from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import math
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed
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
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
try:
    import swanlab  # type: ignore
    SWANLAB_AVAILABLE = True
except Exception:
    SWANLAB_AVAILABLE = False


@dataclass
class Config:
    student_model: str
    teacher_model: str
    output_dir: str
    steps: int = 125
    batch_size: int = 256
    group_size: int = 1
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 1.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    lr_decay: str = "cosine"
    min_lr_ratio: float = 0.1
    save_every: int = 25
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
    grad_accum: int = 1
    eval_every: int = 25
    eval_exact_kl: bool = True
    print_sample: bool = True
    print_every: int = 1
    debug_mask: bool = False
    swanlab_project: str | None = None
    swanlab_name: str | None = None
    swanlab_mode: str = "online"
    # Teacher sharding
    teacher_ds_zero3: bool = False
    teacher_ds_config: str | None = None
    # Micro-batching
    gen_micro_batch: int = 4
    lp_micro_batch: int = 2
    # Progress
    progress: bool = True
    # Fixed weights (0..1); in this variant rKL is fixed to 1.0 during training
    lam_r: float = 1.0
    lam_f: float = 1.0
    # Enable position-decayed fKL weight: pos_ratio = 1 - pos_in_seq / seq_len
    fkl_pos_decay: bool = False
    # Chat formatting
    system_prompt: str | None = "Please reason step by step, and put your final answer within \\boxed{{}}."


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def init_swanlab(cfg: Config) -> None:
    init_kwargs = {"project": cfg.swanlab_project, "config": vars(cfg)}
    if cfg.swanlab_name:
        for name_key in ("experiment_name", "name", "run_name"):
            try:
                swanlab.init(**init_kwargs, **{name_key: cfg.swanlab_name})
                return
            except TypeError:
                continue
    swanlab.init(**init_kwargs)

def lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    decay: str,
) -> float:
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


def load_deepmath_prompts() -> List[str] | None:
    """Load DeepMath-103K questions from HF if available."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        return [row["question"] for row in ds]  # type: ignore
    except Exception:
        return None


def get_prompts(cfg: Config) -> List[str]:
    if cfg.prompts_file:
        return load_prompts(cfg.prompts_file)
    # 支持从本地 HF 数据集目录加载（datasets.save_to_disk 输出）
    if cfg.dataset and os.path.exists(cfg.dataset):
        try:
            from datasets import load_from_disk  # type: ignore

            obj = load_from_disk(cfg.dataset)
            # 可能是 DatasetDict 或 Dataset
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
        except Exception:
            pass
    if cfg.dataset == "deepmath":
        p = load_deepmath_prompts()
        if p:
            return p
    if cfg.dataset == "tulu3":
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
            out: List[str] = []
            for row in ds:  # type: ignore
                msgs = row["messages"]  # type: ignore
                for m in msgs:
                    if m.get("role") == "user":
                        txt = m.get("content", "")
                        if txt:
                            out.append(txt)
                        break
            return out
        except Exception:
            pass
    return load_prompts(None)


def truncate_by_tokens(tok: PreTrainedTokenizerBase, text: str, max_tokens: int) -> str:
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tok.decode(ids)

def apply_chat_format(
    tok: PreTrainedTokenizerBase,
    questions: List[str],
    system_prompt: str | None,
) -> List[str]:
    """Return prompts formatted using the tokenizer chat template.

    Requires tokenizer.apply_chat_template (Qwen3 chat template).
    """
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
    """Return per-position exact KL (no aggregation) with shape [B, T-1].

    kind: "rkl" computes KL(p_s || p_t) = sum_v p_s(v)*(log p_s - log p_t)
          "fkl" computes KL(p_t || p_s) = sum_v p_t(v)*(log p_t - log p_s)
    Align to next-token prediction by dropping last time step.
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

def generate_continuations(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    micro_batch: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, List[int], int]:
    """Generate in micro-batches with total-length cap; return CPU tensor to lower GPU peak.

    Returns:
        seq_std_cpu: torch.LongTensor [B, T] on CPU (right-padded to global max_T)
        plen: List[int] prompt lengths per sample
        pad_id: tokenizer pad id used for padding
    """
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    all_out_raw: List[torch.Tensor] = []
    all_plen: List[int] = []
    max_T = 0
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    with torch.no_grad():
        iterator = range(0, len(prompts), max(1, micro_batch))
        # 移除生成阶段的微批次进度条（统一使用全局训练进度条）
        for i in iterator:
            chunk = prompts[i : i + max(1, micro_batch)]
            batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(device_of(model_for_gen)) for k, v in batch.items()}

            prompt_len = batch["input_ids"].size(1)
            max_new = max(max_tokens - prompt_len, 128)

            gen = model_for_gen.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )
            max_T = max(max_T, gen.size(1))
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

    # Build one CPU tensor to minimize GPU residency
    seq_std = torch.cat([pad_to(t, max_T, pad_id) for t in all_out_raw], dim=0).cpu()
    return seq_std, all_plen, pad_id


def train_step(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    cfg: Config,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
) -> dict:
    # 1) 学生 on-policy 生成（微批），结果存 CPU
    seq_std_cpu, plen_s, pad_id = generate_continuations(
        student, tok, prompts,
        cfg.max_tokens, cfg.temperature, cfg.top_p,
        cfg.gen_micro_batch,
        cfg.progress and accelerator.is_main_process,
    )
    # 2) 构造有效掩码（续写且非 pad）与 token 计数（基于学生序列；教师采样在同一上下文逐位进行）
    B_s, T_s = seq_std_cpu.size()
    am_s_cpu = seq_std_cpu.ne(pad_id)                          # attention mask: 非pad位置为True
    cont_s = torch.zeros((B_s, max(T_s - 1, 0)), dtype=torch.bool)
    for i, L in enumerate(plen_s):
        nonpad = am_s_cpu[i].nonzero()          # [K, 1] 每一行是非padtoken的索引
        if len(nonpad) == 0:
            continue
        first_nonpad = int(nonpad[0].item())        # 第一个非padtoken的索引
        start = max(first_nonpad + L - 1, 0)
        if T_s > 1:
            cont_s[i, start:] = True                # 初始化续写掩码
    valid_s_cpu = cont_s & am_s_cpu[:, 1:]
    tokens_s = int(valid_s_cpu.sum().item())
    if cfg.debug_mask and accelerator.is_main_process and B_s > 0:
        prompt0 = prompts[0]
        prompt_ids = tok(prompt0, return_tensors="pt", truncation=True)["input_ids"][0].cpu()
        prompt_len = int(prompt_ids.numel())
        nonpad = am_s_cpu[0].nonzero()
        if len(nonpad) == 0:
            raise RuntimeError("debug_mask: sample has no non-pad tokens")
        first_nonpad = int(nonpad[0].item())
        seq_len = int(seq_std_cpu.size(1))
        if first_nonpad + prompt_len > seq_len:
            raise RuntimeError(
                f"debug_mask: prompt slice out of range: first_nonpad={first_nonpad}, "
                f"prompt_len={prompt_len}, seq_len={seq_len}"
            )
        seq_prompt = seq_std_cpu[0, first_nonpad : first_nonpad + prompt_len]
        if not torch.equal(seq_prompt.cpu(), prompt_ids):
            raise RuntimeError(
                "debug_mask: prompt ids do not match sequence slice. "
                f"first_nonpad={first_nonpad}, prompt_len={prompt_len}"
            )
        cont_idx = cont_s[0].nonzero()
        if len(cont_idx) == 0:
            raise RuntimeError("debug_mask: continuation mask has no True values")
        first_true = int(cont_idx[0].item())
        expected_start = max(first_nonpad + prompt_len - 1, 0)
        if first_true != expected_start:
            raise RuntimeError(
                "debug_mask: continuation start mismatch. "
                f"first_true={first_true}, expected_start={expected_start}"
            )
    sample_full = ""
    sample_prompt = ""
    sample_cont = ""
    if B_s > 0:
        ids_0 = seq_std_cpu[0]
        nonpad = ids_0.ne(pad_id).nonzero()
        if len(nonpad) > 0:
            first_nonpad = int(nonpad[0].item())
            end = int(nonpad[-1].item() + 1)
            prompt_start = first_nonpad
            prompt_end = max(first_nonpad + plen_s[0], prompt_start)
            sample_full = tok.decode(ids_0[prompt_start:end].tolist())
            if prompt_end > prompt_start:
                sample_prompt = tok.decode(ids_0[prompt_start:prompt_end].tolist())
            if end > prompt_end:
                sample_cont = tok.decode(ids_0[prompt_end:end].tolist())
    if tokens_s == 0:
        return {
            "loss": 0.0,
            "lambda": 0.0,
            "rkl_metric": 0.0,
            "tokens": 0,
            "sample_full": sample_full,
            "sample_prompt": sample_prompt,
            "sample_cont": sample_cont,
        }

    teacher.eval()
    student.train()
    mb = max(1, cfg.lp_micro_batch)

    # 3) 单次前向复用：每个微批只计算一次 student/teacher，再得到 rKL、MC-FKL 与 gating，并立即反向
    eps = 1e-8
    d_rkl_sum = torch.tensor(0.0, device=accelerator.device)
    d_fkl_sum = torch.tensor(0.0, device=accelerator.device)
    tokens_accum = torch.tensor(0.0, device=accelerator.device)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    rkl_loss_sum = torch.tensor(0.0, device=accelerator.device)
    fkl_loss_sum = torch.tensor(0.0, device=accelerator.device)
    # Using fixed lambda; no gating accumulation needed

    for i in range(0, B_s, mb):
        sl = slice(i, i + mb)
        ids_mb = seq_std_cpu[sl].to(accelerator.device, non_blocking=True)
        attn_mb = ids_mb.ne(pad_id).long()
        valid_mb = (cont_s[sl].to(accelerator.device)) & attn_mb[:, 1:].bool()
        with accelerator.autocast():
            with torch.no_grad():
                logits_t = teacher(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
                logp_t = nn.functional.log_softmax(logits_t, dim=-1)
                del logits_t
            logits_s = student(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
            logp_s = nn.functional.log_softmax(logits_s, dim=-1)
            del logits_s
        # rKL-MC（学生 token）
        s_g_s = logp_s[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        t_g_s = logp_t[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        d_rkl_mb = (s_g_s - t_g_s).detach()
        # Policy-gradient form with KL advantage: A = -(logp_s - logp_t)
        rkl_loss_pos = d_rkl_mb * s_g_s

        # fKL-MC：逐位从教师分布采样 token（同一上下文）
        probs_t = logp_t[:, :-1, :].exp()
        Bm, Lm, V = probs_t.shape
        sampled = torch.multinomial(probs_t.reshape(-1, V), num_samples=1).reshape(Bm, Lm)
        t_g_t = logp_t[:, :-1, :].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        s_g_t = logp_s[:, :-1, :].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        # Advantage for fKL: only penalize when teacher prob > student prob
        d_fkl_mb = (t_g_t - s_g_t).detach()
        fkl_loss_pos = - s_g_t
        # Position-decayed weight for fKL/rKL (optional)
        if cfg.fkl_pos_decay:
            # pos_in_seq counts from the first generated token (after prompt)
            # start positions per sample (prompt length - 1)
            starts = torch.tensor([max(L - 1, 0) for L in plen_s[sl]], device=accelerator.device, dtype=torch.long)
            pos_idx = torch.arange(Lm, device=accelerator.device).unsqueeze(0).expand(Bm, Lm) - starts.unsqueeze(1)
            pos_idx = torch.clamp_min(pos_idx, 0)
            seq_len = torch.clamp(Lm - starts, min=1)
            pos_ratio = 1.0 - (pos_idx.float() / seq_len.unsqueeze(1).float())
            pos_ratio = torch.clamp(pos_ratio, 0.0, 1.0)
            rkl_loss_pos = rkl_loss_pos * pos_ratio
            fkl_loss_pos = fkl_loss_pos * pos_ratio

        # Fixed weights (subject to optional decay above)
        lam_R_mb = torch.tensor(float(max(0.0, min(1.0, cfg.lam_r))), device=accelerator.device)
        lam_F_mb = torch.tensor(float(max(0.0, min(1.0, cfg.lam_f))), device=accelerator.device)

        # 汇总损失（按整批学生有效 token 数归一化），并反向
        rkl_loss_mb = (lam_R_mb * rkl_loss_pos).masked_select(valid_mb).sum() / float(max(1, tokens_s))
        fkl_loss_mb = (lam_F_mb * fkl_loss_pos).masked_select(valid_mb).sum() / float(max(1, tokens_s))
        loss_mb = rkl_loss_mb + fkl_loss_mb
        is_last = (i + mb) >= B_s
        if not is_last:
            with accelerator.no_sync(student):
                accelerator.backward(loss_mb)
        else:
            accelerator.backward(loss_mb)
        # 指标与 gating 累计（使用无梯度量）
        d_rkl_sum = d_rkl_sum + d_rkl_mb.masked_select(valid_mb).sum()
        d_fkl_sum = d_fkl_sum + d_fkl_mb.masked_select(valid_mb).sum()
        tokens_accum = tokens_accum + valid_mb.sum()
        loss_sum = loss_sum + loss_mb.detach()
        rkl_loss_sum = rkl_loss_sum + rkl_loss_mb.detach()
        fkl_loss_sum = fkl_loss_sum + fkl_loss_mb.detach()
        del ids_mb, attn_mb, valid_mb, logp_t, logp_s, s_g_s, t_g_s, d_rkl_mb, rkl_loss_pos, probs_t, sampled, t_g_t, s_g_t, fkl_loss_pos, rkl_loss_mb, fkl_loss_mb, loss_mb

    # 跨进程聚合指标（lambda 取 lam_R，rkl_metric 取学生序列上的均值）
    rkl_mean = (
        accelerator.gather_for_metrics(d_rkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum().clamp_min(1)
    ).item()
    fkl_mean = (
        accelerator.gather_for_metrics(d_fkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum().clamp_min(1)
    ).item()
    # 直接报告固定 lambda（此处固定为 1.0）
    lam_value = 1.0
    tokens = int(accelerator.gather_for_metrics(tokens_accum).sum().item())
    loss_val = accelerator.gather_for_metrics(loss_sum).mean().item()
    rkl_loss_val = accelerator.gather_for_metrics(rkl_loss_sum).mean().item()
    fkl_loss_val = accelerator.gather_for_metrics(fkl_loss_sum).mean().item()
    return {
        "loss": float(loss_val),
        "rkl_loss": float(rkl_loss_val),
        "fkl_loss": float(fkl_loss_val),
        "lambda": float(lam_value),
        "rkl_metric": float(rkl_mean),
        "fkl_metric": float(fkl_mean),
        "tokens": tokens,
        "sample_full": sample_full,
        "sample_prompt": sample_prompt,
        "sample_cont": sample_cont,
    }


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mixed = "bf16" if cfg.dtype.lower() == "bf16" else ("fp16" if cfg.dtype.lower() == "fp16" else "no")
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum, mixed_precision=mixed)
    if cfg.swanlab_project and accelerator.is_main_process:
        if not SWANLAB_AVAILABLE:
            raise RuntimeError("swanlab 未安装，请先 pip install swanlab")
        if cfg.swanlab_mode in ("offline", "disabled"):
            os.environ["SWANLAB_MODE"] = cfg.swanlab_mode
        init_swanlab(cfg)

    torch_dtype = torch.bfloat16 if cfg.dtype.lower() == "bf16" else torch.float16 if cfg.dtype.lower() == "fp16" else None
    student = AutoModelForCausalLM.from_pretrained(cfg.student_model, dtype=torch_dtype if torch_dtype else None)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, dtype=torch_dtype if torch_dtype else None)
    tok = AutoTokenizer.from_pretrained(cfg.student_model)
    ensure_pad_token(tok)
    try:
        tok.padding_side = "left"
    except Exception:
        pass

    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未安装，请先 pip install peft")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
        student = get_peft_model(student, lora_cfg)

    optimizer = AdamW(
        student.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    student, optimizer = accelerator.prepare(student, optimizer)
    total_steps = max(1, cfg.steps)
    warmup_steps = cfg.warmup_steps if cfg.warmup_steps > 0 else int(cfg.warmup_ratio * total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))

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
    micro_step = 0
    update_step = 0
    # 全局训练进度条（显示 step/total）
    global_bar = None
    if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
        global_bar = tqdm(total=cfg.steps, desc="train", dynamic_ncols=True)
    acc_loss = 0.0
    acc_rkl_loss = 0.0
    acc_fkl_loss = 0.0
    acc_rkl = 0.0
    acc_fkl = 0.0
    acc_tokens = 0
    acc_count = 0
    while update_step < cfg.steps:
        start = (micro_step * cfg.batch_size) % max(1, len(prompts))
        end = start + cfg.batch_size
        groups = prompts[start:end] if end <= len(prompts) else (prompts[start:] + prompts[: (end % len(prompts))])
        world = accelerator.num_processes
        rank = accelerator.process_index
        groups_shard = [g for i, g in enumerate(groups) if i % max(1, world) == rank]
        batch_prompts = [p for p in groups_shard for _ in range(cfg.group_size)]
        # Apply chat/system formatting if requested
        batch_prompts = apply_chat_format(tok, batch_prompts, cfg.system_prompt)
        if cfg.max_prompt_tokens is not None:
            batch_prompts = [truncate_by_tokens(tok, p, cfg.max_prompt_tokens) for p in batch_prompts]

        with accelerator.accumulate(student):
            next_step = update_step + (1 if accelerator.sync_gradients else 0)
            lr_mult = lr_multiplier(
                next_step,
                total_steps,
                warmup_steps,
                cfg.min_lr_ratio,
                cfg.lr_decay,
            )
            current_lr = cfg.learning_rate * lr_mult
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            metrics = train_step(student, teacher, tok, batch_prompts, cfg, accelerator, optimizer)
            grad_norm = accelerator.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        micro_step += 1

        acc_loss += metrics["loss"]
        acc_rkl_loss += metrics.get("rkl_loss", 0.0)
        acc_fkl_loss += metrics.get("fkl_loss", 0.0)
        acc_rkl += metrics["rkl_metric"] * metrics["tokens"]
        acc_fkl += metrics.get("fkl_metric", 0.0) * metrics["tokens"]
        acc_tokens += metrics["tokens"]
        acc_count += 1
        if accelerator.is_main_process and cfg.print_sample:
            if micro_step % cfg.print_every == 0:
                full_text = metrics.get("sample_full", "")
                prompt_text = metrics.get("sample_prompt", "")
                cont_text = metrics.get("sample_cont", "")
                # if full_text:
                #     accelerator.print("[sample_full]")
                #     accelerator.print(full_text)
                if prompt_text:
                    accelerator.print("[sample_prompt]")
                    accelerator.print(prompt_text)
                if cont_text:
                    accelerator.print("[sample_cont]")
                    accelerator.print(cont_text)

        if accelerator.sync_gradients:
            update_step += 1
            if global_bar is not None:
                global_bar.update(1)

            mean_loss = acc_loss / max(1, acc_count)
            mean_rkl_loss = acc_rkl_loss / max(1, acc_count)
            mean_fkl_loss = acc_fkl_loss / max(1, acc_count)
            mean_rkl = acc_rkl / max(1, acc_tokens)
            mean_fkl = acc_fkl / max(1, acc_tokens)

            if accelerator.is_main_process and (update_step % 10 == 0 or update_step == 1):
                msg = (
                    f"step {update_step:05d}/{cfg.steps:05d} | "
                    f"loss={mean_loss:.4f} rkl={mean_rkl:.4f} d_fkl={mean_fkl:.4f} tokens={acc_tokens}"
                )
                if global_bar is not None:
                    global_bar.set_postfix_str(
                        f"loss={mean_loss:.4f} rkl={mean_rkl:.4f} fkl={mean_fkl:.4f} lr={current_lr:.2e} tok={acc_tokens}"
                    )
                else:
                    accelerator.print(msg)
            if cfg.swanlab_project and accelerator.is_main_process:
                swanlab.log({
                    "train/loss": mean_loss,
                    "train/rkl_loss": mean_rkl_loss,
                    "train/fkl_loss": mean_fkl_loss,
                    "train/reverse_kl": mean_rkl,
                    "train/forward_kl": mean_fkl,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(current_lr),
                    "train/tokens": acc_tokens,
                    "train/step": update_step,
                }, step=update_step)

            acc_loss = 0.0
            acc_rkl_loss = 0.0
            acc_fkl_loss = 0.0
            acc_rkl = 0.0
            acc_fkl = 0.0
            acc_tokens = 0
            acc_count = 0
            # Exact KL eval on a small subset
            if cfg.eval_exact_kl and cfg.eval_every > 0 and update_step % cfg.eval_every == 0:
                with torch.no_grad():
                    k = min(4, len(prompts))
                    eval_prompts = prompts[-k:]
                    eval_prompts = apply_chat_format(tok, eval_prompts, cfg.system_prompt)
                    seqs_cpu, plens, pad_id = generate_continuations(
                        student, tok, eval_prompts,
                        cfg.max_tokens, cfg.temperature, cfg.top_p,
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
                            logits_t = teacher(input_ids=ids_mb, attention_mask=am_mb, use_cache=False).logits
                            logp_s = nn.functional.log_softmax(logits_s, dim=-1)
                            logp_t = nn.functional.log_softmax(logits_t, dim=-1)
                        # per-position exact KL
                        r_pos = per_position_exact_kl(logp_s, logp_t, kind="rkl")
                        f_pos = per_position_exact_kl(logp_s, logp_t, kind="fkl")
                    # valid mask: continuation and non-pad
                    cont_mb = torch.zeros_like(r_pos, dtype=torch.bool)
                    for j, L in enumerate(plens[sl]):
                        nonpad = am_mb[j].nonzero()
                        if len(nonpad) == 0:
                            continue
                        first_nonpad = int(nonpad[0].item())
                        start_j = max(first_nonpad + L - 1, 0)
                        cont_mb[j, start_j:] = True
                        valid_mb = cont_mb & am_mb[:, 1:].bool()
                        r_sum = r_sum + r_pos.masked_select(valid_mb).sum()
                        f_sum = f_sum + f_pos.masked_select(valid_mb).sum()
                        t_sum = t_sum + valid_mb.sum()
                    # aggregate across ranks
                    r_all = accelerator.gather_for_metrics(r_sum).sum()
                    f_all = accelerator.gather_for_metrics(f_sum).sum()
                    t_all = accelerator.gather_for_metrics(t_sum).sum()
                    rkl_exact = (r_all / t_all).item() if t_all.item() > 0 else 0.0
                    fkl_exact = (f_all / t_all).item() if t_all.item() > 0 else 0.0
                    if accelerator.is_main_process:
                        accelerator.print(f"eval rkl_exact={rkl_exact:.4f} fkl_exact={fkl_exact:.4f}")
                    if cfg.swanlab_project and accelerator.is_main_process:
                        swanlab.log({
                            "eval/rkl_exact": rkl_exact,
                            "eval/fkl_exact": fkl_exact,
                            "train/step": update_step,
                        }, step=update_step)

            if accelerator.is_main_process and cfg.save_every > 0 and update_step % cfg.save_every == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"step-{update_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                to_save = accelerator.unwrap_model(student)
                to_save.save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)
                accelerator.print(f"已保存检查点到 {ckpt_dir}")

    if accelerator.is_main_process:
        if global_bar is not None:
            global_bar.close()

        if update_step % cfg.save_every != 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step-{update_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = accelerator.unwrap_model(student)
            to_save.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
        
        accelerator.print(f"训练完成，模型已保存到 {cfg.output_dir}")
        if cfg.swanlab_project and SWANLAB_AVAILABLE:
            finish = getattr(swanlab, "finish", None)
            if callable(finish):
                finish()
    
    # ===== 新增：显式清理分布式进程组 =====
    accelerator.wait_for_everyone()  # 确保所有进程同步
    accelerator.free_memory()  # 释放缓存的内存
    
    # 如果使用了 DeepSpeed，先清理 DeepSpeed
    if cfg.teacher_ds_zero3 and DEEPSPEED_AVAILABLE:
        if hasattr(teacher, 'destroy'):
            teacher.destroy()
    
    # 清理分布式环境
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 最后一次同步
        torch.distributed.destroy_process_group()
    
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Dual-KL 微批反向（正向+反向 KL）- 固定权重版")
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./dual-kl-out")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--group_size", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=2048, help="生成总长度上限（含 prompt）")
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0, help="学习率 warmup 步数（>0 则优先使用）")
    p.add_argument("--warmup_ratio", type=float, default=0.03, help="学习率 warmup 比例（当 warmup_steps=0 时生效）")
    p.add_argument("--lr_decay", type=str, default="cosine", choices=["cosine", "linear", "none"])
    p.add_argument("--min_lr_ratio", type=float, default=0.1, help="decay 最小学习率比例（相对 base lr）")
    # rKL/fKL 均为 MC 实现，无需额外开关
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument(
        "--dataset_field",
        type=str,
        default="question",
        help="当 --dataset 指向本地 HF 数据集目录时，作为用户提示的字段名（默认 question）",
    )
    p.add_argument("--max_prompt_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--no_eval_exact_kl", action="store_true")
    p.add_argument("--no_print_sample", action="store_true")
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--debug_mask", action="store_true")
    p.add_argument("--swanlab_project", type=str, default=None)
    p.add_argument("--swanlab_name", type=str, default=None)
    p.add_argument("--swanlab_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--teacher_ds_zero3", action="store_true")
    p.add_argument("--teacher_ds_config", type=str, default=None)
    p.add_argument("--gen_micro_batch", type=int, default=8)
    p.add_argument("--lp_micro_batch", type=int, default=8)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--lam_r", type=float, default=1.0, help="rKL 权重（固定为1更合理），范围 0..1")
    p.add_argument("--lam_f", type=float, default=1.0, help="fKL 基础权重（叠加位置衰减），范围 0..1")
    p.add_argument("--fkl_pos_decay", action="store_true", help="启用 fKL 的位置衰减权重：pos_ratio = 1 - pos_in_seq/seq_len")
    # Chat formatting
    p.add_argument(
        "--system_prompt",
        type=str,
        default="Please reason step by step, and put your final answer within \\boxed{{}}.",
        help="可选的系统提示（作为 system role 或文本前缀）",
    )
    a = p.parse_args()
    return Config(
        student_model=a.student_model,
        teacher_model=a.teacher_model,
        output_dir=a.output_dir,
        steps=a.steps,
        batch_size=a.batch_size,
        group_size=a.group_size,
        max_tokens=a.max_tokens,
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
        use_lora=a.use_lora,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha if a.lora_alpha is not None else a.lora_r,
        lora_dropout=a.lora_dropout,
        dtype=a.dtype,
        grad_accum=a.grad_accum,
        eval_every=a.eval_every,
        eval_exact_kl=not a.no_eval_exact_kl,
        print_sample=not a.no_print_sample,
        print_every=a.print_every,
        debug_mask=bool(a.debug_mask),
        swanlab_project=a.swanlab_project,
        swanlab_name=a.swanlab_name,
        swanlab_mode=a.swanlab_mode,
        teacher_ds_zero3=a.teacher_ds_zero3,
        teacher_ds_config=a.teacher_ds_config,
        gen_micro_batch=a.gen_micro_batch,
        lp_micro_batch=a.lp_micro_batch,
        progress=not a.no_progress,
        lam_r=max(0.0, min(1.0, a.lam_r)),
        lam_f=max(0.0, min(1.0, a.lam_f)),
        fkl_pos_decay=bool(a.fkl_pos_decay),
        system_prompt=a.system_prompt,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
