#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model and save a standalone model.

Examples:
python merge_lora.py \
--base Qwen/Qwen3-1.7B \
--adapter ./out/opd-1.7b-32b-deepmath_sample32k/step-1000 \
--out ./out/opd-1.7b-32b-deepmath_sample32k-step1000-merged \
--dtype bf16 --device_map auto --safe

python merge_lora.py \
--base Qwen/Qwen3-1.7B \
--adapter ./out/opd-1.7b-32b-deepmath_sample32k/step-400 \
--out ./out/opd-1.7b-32b-deepmath_sample32k-step400-merged \
--dtype bf16 --device_map auto --safe

Requirements:
  pip install -U transformers peft accelerate safetensors

Notes:
  - If GPU memory is limited, set --device_map cpu to merge on CPU.
  - If the base is 4/8-bit quantized, load it in full precision before merging.
"""

import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save.")
    p.add_argument("--base", required=True, help="Base model id or local path (e.g., Qwen/Qwen3-1.7B)")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory (training output)")
    p.add_argument("--out", required=True, help="Output directory for the merged model")
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype when loading the base model (default: bf16)",
    )
    p.add_argument(
        "--device_map",
        choices=["auto", "cpu"],
        default="cpu",
        help="Device map for loading/merging (default: cpu)",
    )
    p.add_argument("--tokenizer_base", default=None, help="Tokenizer source (defaults to --base)")
    p.add_argument("--safe", action="store_true", help="Save with safetensors format")
    return p.parse_args()


def get_dtype(name: str):
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return None


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch_dtype = get_dtype(args.dtype)
    print(f"[info] Loading base: {args.base} (dtype={args.dtype}, device_map={args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype if torch_dtype is not None else None,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )

    print(f"[info] Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[info] Merging LoRA into the base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    tok_src = args.tokenizer_base or args.base
    print(f"[info] Loading tokenizer from: {tok_src}")
    tok = AutoTokenizer.from_pretrained(tok_src)

    print(f"[info] Saving merged model to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=bool(args.safe))
    tok.save_pretrained(args.out)
    print("[ok] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise

