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
    # Dual-KL
    tau: float = 1.0
    alpha: float = 5.0
    gating: str = "soft"  # soft|hard
    rkl: str = "exact"  # exact|mc
    fkl: str = "full"  # full|argmax
    # misc
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


def get_prompts(cfg: Config) -> List[str]:
    if cfg.prompts_file:
        return load_prompts(cfg.prompts_file)
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


def per_position_exact_kl(logp_s: torch.Tensor, logp_t: torch.Tensor, kind: str) -> torch.Tensor:
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


def gather_logp_for_targets(logp: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    target_ids = input_ids[:, 1:]
    return logp[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


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
    seq_std_cpu, plen, pad_id = generate_continuations(
        student, tok, prompts,
        cfg.max_new_tokens, cfg.temperature, cfg.top_p,
        cfg.gen_micro_batch,
        cfg.progress and accelerator.is_main_process,
    )

    B, T = seq_std_cpu.size()
    # 全批有效 mask（CPU 上计算）
    am_full_cpu = seq_std_cpu.ne(pad_id)
    cont_mask = torch.zeros((B, max(T - 1, 0)), dtype=torch.bool)
    for i, L in enumerate(plen):
        start = max(L - 1, 0)
        if T > 1:
            cont_mask[i, start:] = True
    valid_mask_full = cont_mask & am_full_cpu[:, 1:]
    tokens_total = int(valid_mask_full.sum().item())
    if tokens_total == 0:
        return {"loss": 0.0, "lambda": 0.0, "rkl_metric": 0.0, "tokens": 0}

    # 累计指标（仅标量）
    lam_sum = torch.tensor(0.0, device=accelerator.device)
    d_rkl_sum = torch.tensor(0.0, device=accelerator.device)
    tokens_accum = torch.tensor(0.0, device=accelerator.device)
    loss_sum = torch.tensor(0.0, device=accelerator.device)

    teacher.eval()
    student.train()
    mb = max(1, cfg.lp_micro_batch)
    bar_lp = None
    if cfg.progress and _HAVE_TQDM and accelerator.is_main_process:
        bar_lp = tqdm(total=B, desc="lp/back", leave=False, dynamic_ncols=True)

    for i in range(0, B, mb):
        sl = slice(i, i + mb)
        # 将当前切片搬上 GPU
        input_ids_mb = seq_std_cpu[sl].to(accelerator.device, non_blocking=True)
        attn_mb = input_ids_mb.ne(pad_id).long()
        valid_mask_mb = (cont_mask[sl].to(accelerator.device)) & attn_mb[:, 1:].bool()

        # 教师/学生前向（自动混合精度）
        with accelerator.autocast():
            with torch.no_grad():
                logits_t = teacher(input_ids=input_ids_mb, attention_mask=attn_mb, use_cache=True).logits
                logp_t_mb = nn.functional.log_softmax(logits_t, dim=-1)
            logits_s = student(input_ids=input_ids_mb, attention_mask=attn_mb, use_cache=False).logits
            logp_s_mb = nn.functional.log_softmax(logits_s, dim=-1)
        del logits_t, logits_s

        # rKL 指标
        if cfg.rkl == "exact":
            d_rkl_mb = per_position_exact_kl(logp_s_mb, logp_t_mb, kind="rkl")  # [mb,T-1]
            rkl_loss_pos = d_rkl_mb
        elif cfg.rkl == "mc":
            s_g_mb = gather_logp_for_targets(logp_s_mb, input_ids_mb)
            t_g_mb = gather_logp_for_targets(logp_t_mb, input_ids_mb)
            d_rkl_mb = s_g_mb - t_g_mb
            adv_mb = -(s_g_mb - t_g_mb)
            rkl_loss_pos = -adv_mb * s_g_mb
        else:
            raise ValueError("rkl should be exact|mc")

        # FKL 逐位
        if cfg.fkl == "full":
            fkl_loss_pos = -(logp_t_mb[:, :-1, :].exp() * logp_s_mb[:, :-1, :]).sum(dim=-1)
        elif cfg.fkl == "argmax":
            with torch.no_grad():
                tgt_ids_mb = logp_t_mb.argmax(dim=-1)
            fkl_loss_pos = nn.functional.nll_loss(
                logp_s_mb[:, :-1, :].permute(0, 2, 1), tgt_ids_mb[:, 1:], reduction="none"
            )
        else:
            raise ValueError("fkl should be full|argmax")

        # 门控（对 d_rkl 断开梯度，稳定训练）
        if cfg.gating == "soft":
            lam_mb = torch.sigmoid(cfg.alpha * (cfg.tau - d_rkl_mb.detach()))
        elif cfg.gating == "hard":
            lam_mb = (d_rkl_mb.detach() < cfg.tau).float()
        else:
            raise ValueError("gating should be soft|hard")

        loss_pos_mb = lam_mb * rkl_loss_pos + (1.0 - lam_mb) * fkl_loss_pos
        # 归一化到整批 token 数，保持各片梯度等价于整批 mean
        loss_mb = loss_pos_mb.masked_select(valid_mask_mb).sum() / float(tokens_total)

        # 仅最后一片同步 allreduce，降低 DDP 开销
        is_last = (i + mb) >= B
        if not is_last:
            with accelerator.no_sync(student):
                accelerator.backward(loss_mb)
        else:
            accelerator.backward(loss_mb)

        # 指标累计
        lam_sum = lam_sum + lam_mb.masked_select(valid_mask_mb).detach().sum()
        d_rkl_sum = d_rkl_sum + d_rkl_mb.masked_select(valid_mask_mb).detach().sum()
        tokens_accum = tokens_accum + valid_mask_mb.sum().detach()
        loss_sum = loss_sum + loss_mb.detach()

        if bar_lp is not None:
            bar_lp.update(input_ids_mb.size(0))

        # 释放切片张量
        del input_ids_mb, attn_mb, valid_mask_mb, logp_t_mb, logp_s_mb, loss_pos_mb, loss_mb, lam_mb, d_rkl_mb

    if bar_lp is not None:
        bar_lp.close()

    # 跨进程聚合指标
    lam_mean = (accelerator.gather_for_metrics(lam_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum()).item()
    d_rkl_mean = (accelerator.gather_for_metrics(d_rkl_sum).sum() / accelerator.gather_for_metrics(tokens_accum).sum()).item()
    tokens = int(accelerator.gather_for_metrics(tokens_accum).sum().item())
    loss_val = accelerator.gather_for_metrics(loss_sum).mean().item()
    return {"loss": float(loss_val), "lambda": float(lam_mean), "rkl_metric": float(d_rkl_mean), "tokens": tokens}


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
                f"step {step:05d} | loss={metrics['loss']:.4f} λ={metrics['lambda']:.3f} rkl={metrics['rkl_metric']:.4f} tokens={metrics['tokens']}"
            )
        if cfg.wandb_project:
            accelerator.log({
                "train/loss": metrics["loss"],
                "train/lambda": metrics["lambda"],
                "train/rkl_metric": metrics["rkl_metric"],
                "train/tokens": metrics["tokens"],
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
    p = argparse.ArgumentParser(description="Dual-KL 微批反向（正向+反向 KL）")
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
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--gating", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--rkl", type=str, default="exact", choices=["exact", "mc"])
    p.add_argument("--fkl", type=str, default="full", choices=["full", "argmax"])
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
        tau=a.tau,
        alpha=a.alpha,
        gating=a.gating,
        rkl=a.rkl,
        fkl=a.fkl,
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

