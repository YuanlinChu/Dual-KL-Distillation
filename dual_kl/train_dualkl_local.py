from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed
from accelerate import Accelerator
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
    batch_size: int = 2
    group_size: int = 1
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    # 双向 KL 相关
    tau: float = 1.0
    alpha: float = 5.0
    gating: str = "soft"  # soft|hard
    rkl: str = "exact"  # exact|mc
    fkl: str = "full"  # full|argmax
    # 其他
    save_every: int = 100
    prompts_file: str | None = None
    dataset: str | None = None
    max_prompt_tokens: int | None = None
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    dtype: str = "bf16"  # bf16|fp16|fp32
    grad_accum: int = 1
    eval_every: int = 50
    wandb_project: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "online"
    # DeepSpeed options for teacher sharding
    teacher_ds_zero3: bool = False
    teacher_ds_config: str | None = None
    # Micro-batching to reduce peak memory
    gen_micro_batch: int = 8   # generation micro-batch size
    lp_micro_batch: int = 8    # logprob micro-batch size for student/teacher


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
                    txt = m.get("content", "")
                    if txt:
                        prompts.append(txt)
                    break
        return prompts
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
) -> Tuple[torch.Tensor, List[int]]:
    """Generate in micro-batches to cap peak memory."""
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    all_out: List[torch.Tensor] = []
    all_plen: List[int] = []
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
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            pl = batch["input_ids"].ne(tok.pad_token_id).sum(dim=1).tolist()
            all_out.append(gen)
            all_plen.extend(pl)
    return torch.cat(all_out, dim=0), all_plen


def logits_and_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    micro_batch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute logits/logprobs in micro-batches along batch dimension."""
    outs_l: List[torch.Tensor] = []
    outs_lp: List[torch.Tensor] = []
    for i in range(0, input_ids.size(0), max(1, micro_batch)):
        sl = slice(i, i + max(1, micro_batch))
        logits = model(input_ids=input_ids[sl], attention_mask=attention_mask[sl]).logits  # [b,T,V]
        logp = nn.functional.log_softmax(logits, dim=-1)
        outs_l.append(logits)
        outs_lp.append(logp)
    return torch.cat(outs_l, dim=0), torch.cat(outs_lp, dim=0)


def per_position_exact_kl(logp_s: torch.Tensor, logp_t: torch.Tensor, kind: str) -> torch.Tensor:
    """返回每个位置的 KL（不聚合），shape: [B, T-1]
    kind: "rkl"(p_s||p_t) 或 "fkl"(p_t||p_s)。
    计算时统一对齐 next-token 预测（移除最后一位）。
    """
    # 对齐：去掉最后一位的预测，目标为 input_ids[:,1:]
    lps = logp_s[:, :-1, :]  # [B,T-1,V]
    lpt = logp_t[:, :-1, :]
    ps = lps.exp()
    pt = lpt.exp()
    if kind == "rkl":
        # sum p_s * (log p_s - log p_t)
        return (ps * (lps - lpt)).sum(dim=-1)
    elif kind == "fkl":
        # sum p_t * (log p_t - log p_s)
        return (pt * (lpt - lps)).sum(dim=-1)
    else:
        raise ValueError("kind must be rkl or fkl")


def per_position_mc_reverse(student_logp_gather: torch.Tensor, teacher_logp_gather: torch.Tensor) -> torch.Tensor:
    """MC 近似的反向 KL：log p_s - log p_t（按被采样 token），shape [B, T-1]"""
    return student_logp_gather - teacher_logp_gather


def gather_logp_for_targets(logp: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """从 logp[B,T,V] 按目标 token（input_ids[:,1:]）取出 logprob，得到 [B,T-1]"""
    target_ids = input_ids[:, 1:]
    return logp[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def train_step(student: PreTrainedModel, teacher: PreTrainedModel, tok: PreTrainedTokenizerBase, prompts: List[str], cfg: Config, accelerator: Accelerator, optimizer: torch.optim.Optimizer) -> dict:
    # 1) 学生 on-policy 采样
    seq_std, plen = generate_continuations(student, tok, prompts, cfg.max_new_tokens, cfg.temperature, cfg.top_p, cfg.gen_micro_batch)
    am_std = seq_std.ne(tok.pad_token_id).long()

    # 2) 学生与老师在学生序列上的分布
    teacher.eval()
    with torch.no_grad():
        _, logp_t_std = logits_and_logprobs(teacher, seq_std, am_std, cfg.lp_micro_batch)
    student.train()
    logit_s_std, logp_s_std = logits_and_logprobs(student, seq_std, am_std, cfg.lp_micro_batch)

    # 3) 计算 per-position 反向 KL 指标 D_rkl（用于门控）
    if cfg.rkl == "exact":
        d_rkl = per_position_exact_kl(logp_s_std, logp_t_std, kind="rkl")  # [B,T-1]
    elif cfg.rkl == "mc":
        s_g = gather_logp_for_targets(logp_s_std, seq_std)
        t_g = gather_logp_for_targets(logp_t_std, seq_std)
        d_rkl = per_position_mc_reverse(s_g, t_g)
    else:
        raise ValueError("rkl should be exact|mc")

    # 4) 构造续写掩码
    B, Tm1 = d_rkl.shape
    cont_mask = torch.zeros_like(d_rkl, dtype=torch.bool)
    for i, L in enumerate(plen):
        start = max(L - 1, 0)
        cont_mask[i, start:] = True

    # 5) 计算 RKL 与 FKL 的逐位损失
    # RKL：exact 使用 per_position_exact_kl("rkl")；MC 使用 -(A*log p_s) 等价项
    if cfg.rkl == "exact":
        rkl_pos = per_position_exact_kl(logp_s_std, logp_t_std, kind="rkl")  # [B,T-1]
        rkl_loss_pos = rkl_pos  # 已是 KL 值
    else:
        # MC：A = -coef*(log p_s - log p_t)，loss = -A*log p_s = coef*(log p_s - log p_t)*log p_s
        # 这里直接用 -(log p_s - log p_t) * log p_s（不乘 coef，coef 可并入门控/权重），保持与 on-policy 一致的梯度方向
        s_g = gather_logp_for_targets(logp_s_std, seq_std)
        t_g = gather_logp_for_targets(logp_t_std, seq_std)
        adv = -(s_g - t_g)
        rkl_loss_pos = -adv * s_g  # [B,T-1]

    # FKL：full=全词表期望；argmax=老师 argmax token 的 CE
    if cfg.fkl == "full":
        # CE = -sum p_T * log p_S；等价于 FKL 去掉常数项 H(p_T)
        ce_pos = -(logp_t_std.exp() * logp_s_std).sum(dim=-1)  # [B,T-1]
        fkl_loss_pos = ce_pos
    elif cfg.fkl == "argmax":
        with torch.no_grad():
            tgt_ids = logp_t_std.argmax(dim=-1)  # [B,T]
        ce = nn.functional.nll_loss(logp_s_std[:, :-1, :].permute(0, 2, 1), tgt_ids[:, 1:], reduction="none")  # [B,T-1]
        fkl_loss_pos = ce
    else:
        raise ValueError("fkl should be full|argmax")

    # 6) 门控权重 λ_t
    if cfg.gating == "soft":
        lam = torch.sigmoid(cfg.alpha * (cfg.tau - d_rkl))  # [B,T-1]
    elif cfg.gating == "hard":
        lam = (d_rkl < cfg.tau).float()
    else:
        raise ValueError("gating should be soft|hard")

    # 7) 合成逐位损失并聚合
    loss_pos = lam * rkl_loss_pos + (1.0 - lam) * fkl_loss_pos
    loss = loss_pos.masked_select(cont_mask).mean()

    accelerator.backward(loss)

    # 指标（跨进程聚合）
    lam_mean = accelerator.gather_for_metrics(lam.masked_select(cont_mask).detach()).mean().item()
    d_rkl_mean = accelerator.gather_for_metrics(d_rkl.masked_select(cont_mask).detach()).mean().item()
    loss_val = accelerator.gather_for_metrics(loss.detach()).mean().item()
    tokens = accelerator.gather_for_metrics(cont_mask.sum().detach()).sum().item()
    return {"loss": float(loss_val), "lambda": float(lam_mean), "rkl_metric": float(d_rkl_mean), "tokens": int(tokens)}


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mixed = "bf16" if cfg.dtype.lower() == "bf16" else ("fp16" if cfg.dtype.lower() == "fp16" else "no")
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum, mixed_precision=mixed, log_with=["wandb"] if cfg.wandb_project else None)
    if cfg.wandb_project and accelerator.is_main_process:
        if cfg.wandb_mode in ("offline", "disabled"):
            os.environ["WANDB_MODE"] = cfg.wandb_mode
        accelerator.init_trackers(cfg.wandb_project, config=vars(cfg), init_kwargs={"wandb": {"name": cfg.wandb_name}})

    # 模型与分词器
    torch_dtype = torch.bfloat16 if cfg.dtype.lower() == "bf16" else torch.float16 if cfg.dtype.lower() == "fp16" else None
    student = AutoModelForCausalLM.from_pretrained(cfg.student_model, torch_dtype=torch_dtype if torch_dtype else None)
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, torch_dtype=torch_dtype if torch_dtype else None)
    tok = AutoTokenizer.from_pretrained(cfg.student_model)
    # decoder-only models prefer left padding for generation efficiency
    try:
        if getattr(tok, "pad_token_id", None) is not None:
            tok.padding_side = "left"
    except Exception:
        pass
    ensure_pad_token(tok)

    # LoRA（可选）
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft 未安装，请先 pip install peft")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
        student = get_peft_model(student, lora_cfg)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Prepare student with Accelerator; keep teacher out to avoid DDP replication
    student, optimizer = accelerator.prepare(student, optimizer)
    # Teacher: inference only, ZeRO-3 sharded if enabled
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
        if cfg.teacher_ds_config and os.path.exists(cfg.teacher_ds_config):
            import json
            with open(cfg.teacher_ds_config, "r") as f:
                ds_cfg = json.load(f)
        teacher, _, _, _ = deepspeed.initialize(model=teacher, model_parameters=None, config=ds_cfg)

    prompts = get_prompts(cfg)
    step = 0
    while step < cfg.steps:
        # 轮转取 batch，并按进程切分
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
            accelerator.print(f"step {step:05d} | loss={metrics['loss']:.4f} λ={metrics['lambda']:.3f} rkl={metrics['rkl_metric']:.4f} tokens={metrics['tokens']}")
        if cfg.wandb_project:
            accelerator.log({"train/loss": metrics["loss"], "train/lambda": metrics["lambda"], "train/rkl_metric": metrics["rkl_metric"], "train/tokens": metrics["tokens"], "train/step": step}, step=step)

        if accelerator.is_main_process and cfg.eval_every > 0 and step % cfg.eval_every == 0:
            accelerator.print("(提示) 可在此处添加更完整的评估逻辑，例如固定 prompt 子集的 post-KL 估计。")

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
    p = argparse.ArgumentParser(description="Dual-KL 蒸馏（正向+反向 KL）")
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
    # Dual KL
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--gating", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--rkl", type=str, default="exact", choices=["exact", "mc"])
    p.add_argument("--fkl", type=str, default="full", choices=["full", "argmax"])
    # misc
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
    p.add_argument("--teacher_ds_zero3", action="store_true", help="使用 DeepSpeed ZeRO-3 对 teacher 做推理分片")
    p.add_argument("--teacher_ds_config", type=str, default=None, help="DeepSpeed 配置文件（可选）")
    p.add_argument("--gen_micro_batch", type=int, default=8, help="生成阶段的微批大小")
    p.add_argument("--lp_micro_batch", type=int, default=8, help="logprob 前向计算的微批大小")
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
