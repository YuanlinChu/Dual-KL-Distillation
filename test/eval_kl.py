"""
评估两个模型之间的正向KL和反向KL散度。
数据集加载逻辑复用 dualkl_opd_offline.py，取最后20条。

用法:
    python eval_kl.py \
        --model_a /path/to/model_a \
        --model_b /path/to/model_b \
        --dataset /path/to/dataset \
        --dataset_field question
"""

from __future__ import annotations

import argparse
import os
import glob
from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed


# ─── 数据集加载（复用 dualkl_opd_offline.py 逻辑） ───────────────────────────


def load_prompts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_prompts(
    dataset: str | None,
    dataset_field: str,
    prompts_file: str | None,
) -> List[str]:
    if prompts_file:
        return load_prompts_from_file(prompts_file)

    if dataset and os.path.exists(dataset):
        # 方式1: load_from_disk（Arrow 格式）
        try:
            from datasets import load_from_disk

            obj = load_from_disk(dataset)
            if hasattr(obj, "keys"):
                split_name = "train" if "train" in obj.keys() else list(obj.keys())[0]
                ds = obj[split_name]
            else:
                ds = obj
            col = dataset_field or "question"
            if col in ds.column_names:
                return [str(v) for v in ds[col]]
            for alt in ["question", "prompt", "input", "text"]:
                if alt in ds.column_names:
                    return [str(v) for v in ds[alt]]
        except Exception:
            pass

        # 方式2: 直接读取 parquet 文件（兼容 HF repo 格式 data/*.parquet）
        try:
            import pandas as pd

            parquet_files = sorted(glob.glob(os.path.join(dataset, "data", "*.parquet")))
            if not parquet_files:
                parquet_files = sorted(glob.glob(os.path.join(dataset, "*.parquet")))
            if parquet_files:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
                col = dataset_field or "question"
                if col in df.columns:
                    return df[col].dropna().astype(str).tolist()
                for alt in ["question", "prompt", "input", "text"]:
                    if alt in df.columns:
                        return df[alt].dropna().astype(str).tolist()
        except Exception:
            pass

    raise ValueError("无法加载数据集，请检查 --dataset 或 --prompts_file 参数")


# ─── Chat 格式化 ─────────────────────────────────────────────────────────────


def apply_chat_format(
    tok: PreTrainedTokenizerBase,
    questions: List[str],
    system_prompt: str | None,
) -> List[str]:
    if not (hasattr(tok, "apply_chat_template") and callable(getattr(tok, "apply_chat_template"))):
        raise RuntimeError("Tokenizer 不支持 apply_chat_template")
    out: List[str] = []
    for q in questions:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": q})
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


# ─── KL 计算 ─────────────────────────────────────────────────────────────────


def per_position_exact_kl(logp_a: torch.Tensor, logp_b: torch.Tensor, kind: str) -> torch.Tensor:
    """逐位置精确 KL 散度，shape [B, T-1]。

    kind:
        "rkl" -> KL(p_a || p_b) = sum_v p_a(v) * (log p_a - log p_b)  (反向KL)
        "fkl" -> KL(p_b || p_a) = sum_v p_b(v) * (log p_b - log p_a)  (正向KL)
    """
    lpa = logp_a[:, :-1, :]
    lpb = logp_b[:, :-1, :]
    pa = lpa.exp()
    pb = lpb.exp()
    if kind == "rkl":
        return (pa * (lpa - lpb)).sum(dim=-1)
    elif kind == "fkl":
        return (pb * (lpb - lpa)).sum(dim=-1)
    else:
        raise ValueError("kind must be 'rkl' or 'fkl'")


# ─── 生成续写 ────────────────────────────────────────────────────────────────


def generate_continuations(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    micro_batch: int,
) -> Tuple[torch.Tensor, List[int], int]:
    """用学生模型生成续写，返回 (seq_cpu, prompt_lengths, pad_id)。"""
    model.eval()
    all_out: List[torch.Tensor] = []
    all_plen: List[int] = []
    max_T = 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(prompts), max(1, micro_batch)):
            chunk = prompts[i : i + max(1, micro_batch)]
            batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            prompt_len = batch["input_ids"].size(1)
            max_new = max(max_tokens - prompt_len, 128)

            gen = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
            )
            max_T = max(max_T, gen.size(1))
            all_out.append(gen.detach())
            all_plen.extend(batch["input_ids"].ne(pad_id).sum(dim=1).tolist())

    def pad_to(tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
        if tensor.size(1) >= target_len:
            return tensor[:, :target_len]
        pad_cols = target_len - tensor.size(1)
        return torch.cat([tensor, torch.full((tensor.size(0), pad_cols), pad_val, dtype=tensor.dtype, device=tensor.device)], dim=1)

    seq_cpu = torch.cat([pad_to(t, max_T, pad_id) for t in all_out], dim=0).cpu()
    return seq_cpu, all_plen, pad_id


# ─── 在给定续写序列上计算 KL ─────────────────────────────────────────────────

@torch.no_grad()
def compute_kl_on_sequences(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    seqs_cpu: torch.Tensor,
    plens: List[int],
    pad_id: int,
    micro_batch_logp: int,
    device: torch.device,
) -> dict:
    """在给定的续写序列上计算两个模型之间的正向/反向 KL。"""
    B, T = seqs_cpu.shape
    rkl_sum = torch.tensor(0.0, device=device)
    fkl_sum = torch.tensor(0.0, device=device)
    token_count = torch.tensor(0.0, device=device)

    for i in range(0, B, max(1, micro_batch_logp)):
        ids_mb = seqs_cpu[i : i + micro_batch_logp].to(device)
        attn_mb = ids_mb.ne(pad_id).long()

        logits_a = model_a(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
        logits_b = model_b(input_ids=ids_mb, attention_mask=attn_mb, use_cache=False).logits
        logp_a = nn.functional.log_softmax(logits_a, dim=-1)
        logp_b = nn.functional.log_softmax(logits_b, dim=-1)
        del logits_a, logits_b

        rkl_pos = per_position_exact_kl(logp_a, logp_b, kind="rkl")  # KL(a || b)
        fkl_pos = per_position_exact_kl(logp_a, logp_b, kind="fkl")  # KL(b || a)

        # 构造续写掩码：只在 prompt 之后的非 pad 位置计算
        cont_mask = torch.zeros_like(rkl_pos, dtype=torch.bool)
        for j in range(ids_mb.size(0)):
            idx = i + j
            if idx >= len(plens):
                break
            nonpad = attn_mb[j].nonzero()
            if len(nonpad) == 0:
                continue
            first_nonpad = int(nonpad[0].item())
            start = max(first_nonpad + plens[idx] - 1, 0)
            cont_mask[j, start:] = True

        valid_mask = cont_mask & attn_mb[:, 1:].bool()
        rkl_sum += rkl_pos.masked_select(valid_mask).sum()
        fkl_sum += fkl_pos.masked_select(valid_mask).sum()
        token_count += valid_mask.sum()

        del ids_mb, attn_mb, logp_a, logp_b

    total_tokens = int(token_count.item())
    rkl_mean = (rkl_sum / token_count).item() if total_tokens > 0 else 0.0
    fkl_mean = (fkl_sum / token_count).item() if total_tokens > 0 else 0.0

    return {
        "reverse_kl": rkl_mean,   # KL(model_a || model_b)
        "forward_kl": fkl_mean,   # KL(model_b || model_a)
        "total_tokens": total_tokens,
    }

# ─── 主评估流程 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_kl(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    micro_batch_gen: int,
    micro_batch_logp: int,
    system_prompt: str | None,
    device: torch.device,
) -> dict:
    formatted_prompts = apply_chat_format(tok, prompts, system_prompt)

    # ── model_a 续写 ──
    print(f"[eval] model_a 生成续写中... ({len(formatted_prompts)} 条)")
    seqs_a_cpu, plens_a, pad_id = generate_continuations(
        model_a, tok, formatted_prompts, max_tokens, temperature, micro_batch_gen,
    )
    print(f"[eval] model_a 续写完成，序列形状: {seqs_a_cpu.shape}")

    print("[eval] 在 model_a 续写上计算 KL...")
    results_on_a = compute_kl_on_sequences(
        model_a, model_b, seqs_a_cpu, plens_a, pad_id, micro_batch_logp, device,
    )

    # ── model_b 续写 ──
    print(f"[eval] model_b 生成续写中... ({len(formatted_prompts)} 条)")
    seqs_b_cpu, plens_b, pad_id_b = generate_continuations(
        model_b, tok, formatted_prompts, max_tokens, temperature, micro_batch_gen,
    )
    print(f"[eval] model_b 续写完成，序列形状: {seqs_b_cpu.shape}")

    print("[eval] 在 model_b 续写上计算 KL...")
    results_on_b = compute_kl_on_sequences(
        model_a, model_b, seqs_b_cpu, plens_b, pad_id_b, micro_batch_logp, device,
    )

    return {
        "on_model_a_gen": results_on_a,
        "on_model_b_gen": results_on_b,
        "num_prompts": len(prompts),
    }

def main():
    parser = argparse.ArgumentParser(description="评估两个模型之间的正向/反向 KL 散度")
    parser.add_argument("--model_a", type=str, help="模型 A 路径（也用于生成续写）", default="/data/oss_bucket_0/zhulin/models/Qwen3-1.7B-Base")
    # parser.add_argument("--model_a", type=str, help="模型 A 路径（也用于生成续写）", default="/data/oss_bucket_0/zhulin/output/Qwen3-1.7B-Base-sft-checkpoint-79")
    # parser.add_argument("--model_a", type=str, help="模型 A 路径（也用于生成续写）", default="/data/oss_bucket_0/zhulin/output/opd-out/dkl-1.7b_base_sft_80-8b-r1f1/step-50")
    parser.add_argument("--model_b", type=str, help="模型 B 路径", default="/data/oss_bucket_0/zhulin/models/Qwen3-8B")
    parser.add_argument("--dataset", type=str, help="数据集路径（本地 HF 目录或 parquet）", default="/data/oss_bucket_0/zhulin/datasets/DeepMath-103K")
    parser.add_argument("--prompts_file", type=str, default=None, help="纯文本 prompts 文件")
    parser.add_argument("--dataset_field", type=str, default="question", help="数据集中 prompt 字段名")
    parser.add_argument("--num_eval", type=int, default=20, help="取最后 N 条数据评估")
    parser.add_argument("--max_tokens", type=int, default=2048, help="生成总长度上限（含 prompt）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gen_micro_batch", type=int, default=4, help="生成阶段微批大小")
    parser.add_argument("--lp_micro_batch", type=int, default=2, help="前向计算阶段微批大小")
    parser.add_argument("--system_prompt", type=str,
                        default="Please reason step by step, and put your final answer within \\boxed{{}}.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
    }[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集，取最后 N 条
    all_prompts = get_prompts(args.dataset, args.dataset_field, args.prompts_file)
    eval_prompts = all_prompts[-args.num_eval:]
    print(f"[info] 数据集共 {len(all_prompts)} 条，取最后 {len(eval_prompts)} 条评估")

    # 加载模型
    print(f"[info] 加载 model_a: {args.model_a}")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a, torch_dtype=torch_dtype if torch_dtype else None,
    ).to(device).eval()

    print(f"[info] 加载 model_b: {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b, torch_dtype=torch_dtype if torch_dtype else None,
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.model_a)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass

    # 评估
    results = evaluate_kl(
        model_a=model_a,
        model_b=model_b,
        tok=tok,
        prompts=eval_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        micro_batch_gen=args.gen_micro_batch,
        micro_batch_logp=args.lp_micro_batch,
        system_prompt=args.system_prompt,
        device=device,
    )

    # 输出结果
    res_a = results["on_model_a_gen"]
    res_b = results["on_model_b_gen"]

    print("\n" + "=" * 60)
    print("KL Divergence Evaluation Results")
    print("=" * 60)
    print(f"  model_a:      {args.model_a}")
    print(f"  model_b:      {args.model_b}")
    print(f"  num_prompts:  {results['num_prompts']}")
    print()
    print("── On model_a generations ──")
    print(f"  tokens:       {res_a['total_tokens']}")
    print(f"  reverse_kl:   {res_a['reverse_kl']:.6f}  (KL(model_a || model_b))")
    print(f"  forward_kl:   {res_a['forward_kl']:.6f}  (KL(model_b || model_a))")
    print()
    print("── On model_b generations ──")
    print(f"  tokens:       {res_b['total_tokens']}")
    print(f"  reverse_kl:   {res_b['reverse_kl']:.6f}  (KL(model_a || model_b))")
    print(f"  forward_kl:   {res_b['forward_kl']:.6f}  (KL(model_b || model_a))")
    print("=" * 60)


if __name__ == "__main__":
    main()