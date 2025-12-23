from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
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


@dataclass
class Config:
    student_model: str
    teacher_model: str
    output_dir: str
    steps: int = 1000
    batch_size: int = 2
    group_size: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    save_every: int = 100
    prompts_file: str | None = None
    dataset: str | None = None
    max_prompt_tokens: int | None = None
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    dtype: str = "bf16"
    grad_accum: int = 1
    eval_every: int = 50
    wandb_project: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "online"
    # Teacher sharding
    teacher_ds_zero3: bool = False
    teacher_ds_config: str | None = None
    # Micro-batching
    gen_micro_batch: int = 8
    lp_micro_batch: int = 8
    # Progress
    progress: bool = True
    # Fixed weights (0..1) for rKL and fKL
    lam_r: float = 0.5
    lam_f: float = 0.5


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def load_prompts(path: str | None) -> List[str]:
    if path is None or not os.path.exists(path):
        return [
            "解释熵的直观含义。",
            "写一首四句的小诗，主题是海。",
            "单元测试的优缺点有哪些？",
            "简要对比 HTTP/1.1 与 HTTP/2 的差别。",
        ]
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
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    micro_batch: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, List[int], int]:
    """Generate in micro-batches; return CPU tensor to lower GPU peak.

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
        bar = None
        if show_progress and _HAVE_TQDM:
            bar = tqdm(total=len(prompts), desc="gen", leave=False, dynamic_ncols=True)
        for i in iterator:
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
            max_T = max(max_T, gen.size(1))
            all_out_raw.append(gen.detach())
            all_plen.extend(batch["input_ids"].ne(pad_id).sum(dim=1).tolist())
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()

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
        cfg.max_new_tokens, cfg.temperature, cfg.top_p,
        cfg.gen_micro_batch,
        cfg.progress and accelerator.is_main_process,
    )
    # 2) 构造有效掩码（续写且非 pad）与 token 计数（基于学生序列；教师采样在同一上下文逐位进行）
    B_s, T_s = seq_std_cpu.size()
    am_s_cpu = seq_std_cpu.ne(pad_id)
    cont_s = torch.zeros((B_s, max(T_s - 1, 0)), dtype=torch.bool)
    for i, L in enumerate(plen_s):
        start = max(L - 1, 0)
        if T_s > 1:
            cont_s[i, start:] = True
    valid_s_cpu = cont_s & am_s_cpu[:, 1:]
    tokens_s = int(valid_s_cpu.sum().item())
    if tokens_s == 0:
        return {"loss": 0.0, "lambda": 0.0, "rkl_metric": 0.0, "tokens": 0}

    teacher.eval()
    student.train()
    mb = max(1, cfg.lp_micro_batch)

    # 3) 单次前向复用：每个微批只计算一次 student/teacher，再得到 rKL、MC-FKL 与 gating，并立即反向
    eps = 1e-8
    d_rkl_sum = torch.tensor(0.0, device=accelerator.device)
    d_fkl_sum = torch.tensor(0.0, device=accelerator.device)
    tokens_accum = torch.tensor(0.0, device=accelerator.device)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    # Using fixed lambda; no gating accumulation needed

    for i in range(0, B_s, mb):
        sl = slice(i, i + mb)
        ids_mb = seq_std_cpu[sl].to(accelerator.device, non_blocking=True)
        attn_mb = ids_mb.ne(pad_id).long()
        valid_mb = (cont_s[sl].to(accelerator.device)) & attn_mb[:, 1:].bool()
        with accelerator.autocast():
            with torch.no_grad():
                logits_t = teacher(input_ids=ids_mb, attention_mask=attn_mb, use_cache=True).logits
                logp_t = nn.functional.log_softmax(logits_t, dim=-1)
                del logits_t
            logits_s = student(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
            logp_s = nn.functional.log_softmax(logits_s, dim=-1)
            del logits_s
        # rKL-MC（学生 token）
        s_g_s = logp_s[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        t_g_s = logp_t[:, :-1, :].gather(-1, ids_mb[:, 1:].unsqueeze(-1)).squeeze(-1)
        d_rkl_mb = (s_g_s - t_g_s).detach()
        rkl_loss_pos = d_rkl_mb * s_g_s

        # fKL-MC：逐位从教师分布采样 token（同一上下文）
        probs_t = logp_t[:, :-1, :].exp()
        Bm, Lm, V = probs_t.shape
        sampled = torch.multinomial(probs_t.reshape(-1, V), num_samples=1).reshape(Bm, Lm)
        t_g_t = logp_t[:, :-1, :].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        s_g_t = logp_s[:, :-1, :].gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        d_fkl_mb = (t_g_t - s_g_t).detach()
        fkl_loss_pos = - s_g_t
        # 使用固定权重（由传参提供，范围 0..1，不再进行归一化）
        lam_r_val = float(max(0.0, min(1.0, cfg.lam_r)))
        lam_f_val = float(max(0.0, min(1.0, cfg.lam_f)))
        lam_R_mb = torch.tensor(lam_r_val, device=accelerator.device)
        lam_F_mb = torch.tensor(lam_f_val, device=accelerator.device)

        # 汇总损失（按整批学生有效 token 数归一化），并反向
        loss_pos_mb = lam_R_mb * rkl_loss_pos + lam_F_mb * fkl_loss_pos
        loss_mb = loss_pos_mb.masked_select(valid_mb).sum() / float(max(1, tokens_s))
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
        del ids_mb, attn_mb, valid_mb, logp_t, logp_s, s_g_s, t_g_s, d_rkl_mb, rkl_loss_pos, probs_t, sampled, t_g_t, s_g_t, fkl_loss_pos, loss_mb

    # 跨进程聚合指标（lambda 取 lam_R，rkl_metric 取学生序列上的均值）
    rkl_mean = (
        accelerator.gather_for_metrics(d_rkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum().clamp_min(1)
    ).item()
    fkl_mean = (
        accelerator.gather_for_metrics(d_fkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum().clamp_min(1)
    ).item()
    # 直接报告固定 lambda（这里返回 lam_r 作为 "lambda" 以保持下游日志兼容）
    lam_value = float(max(0.0, min(1.0, cfg.lam_r)))
    tokens = int(accelerator.gather_for_metrics(tokens_accum).sum().item())
    loss_val = accelerator.gather_for_metrics(loss_sum).mean().item()
    return {"loss": float(loss_val), "lambda": float(lam_value), "rkl_metric": float(rkl_mean), "fkl_metric": float(fkl_mean), "tokens": tokens}


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mixed = "bf16" if cfg.dtype.lower() == "bf16" else ("fp16" if cfg.dtype.lower() == "fp16" else "no")
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum, mixed_precision=mixed, log_with=["wandb"] if cfg.wandb_project else None)
    if cfg.wandb_project and accelerator.is_main_process:
        if cfg.wandb_mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = cfg.wandb_mode
        accelerator.init_trackers(cfg.wandb_project, config=vars(cfg), init_kwargs={"wandb": {"name": cfg.wandb_name}})

    torch_dtype = torch.bfloat16 if cfg.dtype.lower() == "bf16" else torch.float16 if cfg.dtype.lower() == "fp16" else None
    student = AutoModelForCausalLM.from_pretrained(cfg.student_model, dtype=torch_dtype if torch_dtype else None)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, dtype=torch_dtype if torch_dtype else None)
    tok = AutoTokenizer.from_pretrained(cfg.student_model)
    try:
        if getattr(tok, "pad_token_id", None) is not None:
            tok.padding_side = "left"
    except Exception:
        pass
    ensure_pad_token(tok)

    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未安装，请先 pip install peft")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
        student = get_peft_model(student, lora_cfg)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
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
    step = 0
    while step < cfg.steps:
        start = (step * cfg.batch_size) % max(1, len(prompts))
        end = start + cfg.batch_size
        groups = prompts[start:end] if end <= len(prompts) else (prompts[start:] + prompts[: (end % len(prompts))])
        world = accelerator.num_processes
        rank = accelerator.process_index
        groups_shard = [g for i, g in enumerate(groups) if i % max(1, world) == rank]
        batch_prompts = [p for p in groups_shard for _ in range(cfg.group_size)]
        if cfg.max_prompt_tokens is not None:
            batch_prompts = [truncate_by_tokens(tok, p, cfg.max_prompt_tokens) for p in batch_prompts]

        with accelerator.accumulate(student):
            optimizer.zero_grad(set_to_none=True)
            metrics = train_step(student, teacher, tok, batch_prompts, cfg, accelerator, optimizer)
            accelerator.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1

        if accelerator.is_main_process and (step % 10 == 0 or step == 1):
            accelerator.print(
                f"step {step:05d} | loss={metrics['loss']:.4f} λ={metrics['lambda']:.3f} rkl={metrics['rkl_metric']:.4f} fkl={metrics.get('fkl_metric', 0.0):.4f} tokens={metrics['tokens']}"
            )
        if cfg.wandb_project:
            accelerator.log({
                "train/loss": metrics["loss"],
                "train/lambda": metrics["lambda"],
                "train/rkl_metric": metrics["rkl_metric"],
                "train/fkl_metric": metrics.get("fkl_metric", 0.0),
                "train/tokens": metrics["tokens"],
                "train/step": step,
            }, step=step)

        # Exact KL eval on a small subset
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            with torch.no_grad():
                k = min(4, len(prompts))
                eval_prompts = prompts[-k:]
                seqs_cpu, plens, pad_id = generate_continuations(
                    student, tok, eval_prompts,
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
                    # per-position exact KL
                    r_pos = per_position_exact_kl(logp_s, logp_t, kind="rkl")
                    f_pos = per_position_exact_kl(logp_s, logp_t, kind="fkl")
                    # valid mask: continuation and non-pad
                    cont_mb = torch.zeros_like(r_pos, dtype=torch.bool)
                    for j, L in enumerate(plens[sl]):
                        start_j = max(L - 1, 0)
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
                if cfg.wandb_project:
                    accelerator.log({
                        "eval/rkl_exact": rkl_exact,
                        "eval/fkl_exact": fkl_exact,
                        "train/step": step,
                    }, step=step)

        if accelerator.is_main_process and cfg.save_every > 0 and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = accelerator.unwrap_model(student)
            to_save.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            accelerator.print(f"已保存检查点到 {ckpt_dir}")

    if accelerator.is_main_process:
        to_save = accelerator.unwrap_model(student)
        to_save.save_pretrained(cfg.output_dir)
        tok.save_pretrained(cfg.output_dir)
        accelerator.print(f"训练完成，模型已保存到 {cfg.output_dir}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Dual-KL 微批反向（正向+反向 KL）- 固定权重版")
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./dual-kl-out")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--group_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    # rKL/fKL 均为 MC 实现，无需额外开关
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--teacher_ds_zero3", action="store_true")
    p.add_argument("--teacher_ds_config", type=str, default=None)
    p.add_argument("--gen_micro_batch", type=int, default=8)
    p.add_argument("--lp_micro_batch", type=int, default=8)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--lam_r", type=float, default=0.5, help="rKL 权重，范围 0..1")
    p.add_argument("--lam_f", type=float, default=0.5, help="fKL 权重，范围 0..1")
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
        save_every=a.save_every,
        prompts_file=a.prompts_file,
        dataset=a.dataset,
        max_prompt_tokens=a.max_prompt_tokens,
        seed=a.seed,
        use_lora=a.use_lora,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        dtype=a.dtype,
        grad_accum=a.grad_accum,
        eval_every=a.eval_every,
        wandb_project=a.wandb_project,
        wandb_name=a.wandb_name,
        wandb_mode=a.wandb_mode,
        teacher_ds_zero3=a.teacher_ds_zero3,
        teacher_ds_config=a.teacher_ds_config,
        gen_micro_batch=a.gen_micro_batch,
        lp_micro_batch=a.lp_micro_batch,
        progress=not a.no_progress,
        lam_r=max(0.0, min(1.0, a.lam_r)),
        lam_f=max(0.0, min(1.0, a.lam_f)),
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
