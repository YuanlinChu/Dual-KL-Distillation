from __future__ import annotations

import argparse
import json
from typing import Optional

from .tasks import (
    EvalConfig,
    eval_gsm8k,
    eval_math,
    eval_aime_jsonl,
    eval_aime_hf,
    eval_math500_hf,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified eval runner")
    p.add_argument("--task", type=str, required=True, choices=["gsm8k", "math", "math500", "aime", "aime_jsonl"], help="Which eval to run")
    p.add_argument("--model", type=str, required=True, help="HF model id or local path (or LoRA adapter dir)")
    p.add_argument("--base_model", type=str, default=None, help="Base model id when --model is a LoRA adapter dir")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation dtype")
    p.add_argument("--n_samples", type=int, default=None, help="Limit number of samples for quick eval")
    p.add_argument("--batch_size", type=int, default=8, help="Generation micro-batch size")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--save_outputs", type=str, default=None, help="Optional JSONL path to save per-sample outputs")
    p.add_argument("--jsonl_file", type=str, default=None, help="For task=aime_jsonl, path to {question,answer} JSONL")
    p.add_argument("--aime_split", type=str, default="test", help="AIME split (e.g., test|dev) for HuggingFaceH4/aime_2024")
    p.add_argument("--math500_split", type=str, default="test", help="MATH-500 split (e.g., test) for HuggingFaceH4/MATH-500")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        model=args.model,
        base_model=args.base_model,
        dtype=args.dtype,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        save_outputs=args.save_outputs,
    )

    if args.task == "gsm8k":
        res = eval_gsm8k(cfg)
    elif args.task == "math":
        res = eval_math(cfg)
    elif args.task == "aime_jsonl":
        if not args.jsonl_file:
            raise ValueError("--jsonl_file is required for task=aime_jsonl")
        res = eval_aime_jsonl(cfg, args.jsonl_file)
    elif args.task == "aime":
        res = eval_aime_hf(cfg, split=args.aime_split)
    elif args.task == "math500":
        res = eval_math500_hf(cfg, split=args.math500_split)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
