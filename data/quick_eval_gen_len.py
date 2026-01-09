#!/usr/bin/env python3
"""
Quickly run generation on ~N samples and report average output length.

Supports:
- Model: any HF CausalLM (e.g., Qwen/Qwen3-32B)
- Dataset: HF hub ID (source[,config,split]) or local directory saved via datasets.save_to_disk
- Field: which column to read prompts from (default: question)

Examples:
python data/quick_eval_gen_len.py \
    --model Qwen/Qwen3-32B \
    --dataset data/DeepMath-32k \
    --field question \
    --limit 16 \
    --batch_size 16 \
    --max_new_tokens 2048 \
    --do_sample \
    --dtype bf16 \
    --device_map auto \
    --trust_remote_code \
    --save_jsonl data/qwen32b_deepmath32k_16.jsonl

python data/quick_eval_gen_len.py \
    --model out/dkl-1.7b-32b-deepmath_sample32k-step400-merged \
    --dataset data/DeepMath-32k \
    --field question \
    --limit 16 \
    --batch_size 16 \
    --max_new_tokens 2048 \
    --do_sample \
    --dtype bf16 \
    --device_map auto \
    --trust_remote_code \
    --save_jsonl data/1.7b-dkl-posdecay-400_deepmath32k_16_2048.jsonl

python data/quick_eval_gen_len.py \
    --model out/opd-1.7b-32b-deepmath_sample32k-step400-merged \
    --dataset data/DeepMath-32k \
    --field question \
    --limit 16 \
    --batch_size 16 \
    --max_new_tokens 4096 \
    --do_sample \
    --dtype bf16 \
    --device_map auto \
    --trust_remote_code \
    --save_jsonl data/1.7b-opd-400_deepmath32k_16.jsonl

Requirements:
  pip install -U transformers datasets accelerate safetensors

Save outputs:
  Add --save_jsonl path to write per-sample results as JSONL with fields:
    {"idx", "prompt", "generated", "gen_len_tokens", "gen_len_chars"}
"""

import argparse
import json
import os
import random
from typing import Iterable, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ~N samples and measure average output length")
    # Model args
    p.add_argument("--model", required=True, help="Model id or local path (e.g., Qwen/Qwen3-32B)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype")
    p.add_argument("--device_map", default="auto", choices=["auto", "cpu"], help="Device mapping for loading")
    p.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code when loading model/tokenizer")

    # Data args
    p.add_argument("--dataset", required=True, help="HF dataset id or local load_from_disk path")
    p.add_argument("--split", default="train", help="Split when using HF hub dataset (default: train)")
    p.add_argument("--field", default="question", help="Text field/column name to read prompts from")
    p.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    p.add_argument("--max_prompt_tokens", type=int, default=None, help="Optional truncation of prompt tokens")

    # Generation args
    p.add_argument("--batch_size", type=int, default=1, help="Per-step batch size for generation")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (default off => greedy)")

    # Output control
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    p.add_argument("--save_jsonl", default=None, help="Optional path to save per-sample outputs as JSONL")

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


def ensure_pad_token(tok):
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token


def load_prompts(dataset: str, split: str, field: str, limit: int, seed: int) -> List[str]:
    # Local path → load_from_disk
    if os.path.exists(dataset):
        from datasets import load_from_disk  # local import

        obj = load_from_disk(dataset)
        if hasattr(obj, "keys"):
            split_name = "train" if "train" in obj.keys() else list(obj.keys())[0]
            ds = obj[split_name]
        else:
            ds = obj
    else:
        # Remote hub dataset
        ds = load_dataset(dataset, split=split)

    # Column resolution
    col = field
    if col not in ds.column_names:
        for alt in ["question", "prompt", "input", "text"]:
            if alt in ds.column_names:
                col = alt
                break

    # Shuffle then take first N for randomness and reproducibility
    ds = ds.shuffle(seed=seed)
    n = min(limit, len(ds))
    ds = ds.select(range(n))
    return [str(v) for v in ds[col]]


def chunked(it: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(it), max(1, size)):
        yield it[i : i + max(1, size)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Load model/tokenizer
    torch_dtype = get_dtype(args.dtype)
    print(f"[info] Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tok)
    # For generation with padding, left padding often works better
    try:
        if getattr(tok, "pad_token_id", None) is not None:
            tok.padding_side = "left"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if torch_dtype is not None else None,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Load prompts
    prompts = load_prompts(args.dataset, args.split, args.field, args.limit, args.seed)
    if len(prompts) == 0:
        print("[warn] No prompts loaded. Check dataset path/split/field.")
        return
    print(f"[info] Loaded {len(prompts)} prompts from {args.dataset}")

    # Optional truncation of prompts by tokens
    if args.max_prompt_tokens is not None:
        trunc_prompts: List[str] = []
        for p in prompts:
            ids = tok.encode(p)
            if len(ids) > args.max_prompt_tokens:
                ids = ids[: args.max_prompt_tokens]
                trunc_prompts.append(tok.decode(ids))
            else:
                trunc_prompts.append(p)
        prompts = trunc_prompts

    total_gen_tokens = 0
    total_gen_chars = 0
    results = []  # for optional saving

    with torch.no_grad():
        bar = None
        if not args.no_progress and _HAVE_TQDM:
            bar = tqdm(total=len(prompts), desc="gen", dynamic_ncols=True)
        for chunk in chunked(prompts, args.batch_size):
            batch = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            batch = {k: v.to(model.device) for k, v in batch.items()}

            gen_out = model.generate(
                **batch,
                do_sample=bool(args.do_sample),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            # Compute lengths per item (true generated length, excluding right padding)
            # - We count non-PAD tokens after the prompt. Note: in many decoder-only models
            #   PAD == EOS; this matches training where EOS is treated as PAD for token counting.
            prompt_lens = batch["input_ids"].ne(tok.pad_token_id).sum(dim=1).tolist()
            for i in range(gen_out.size(0)):
                out_ids = gen_out[i]
                p_len = int(prompt_lens[i])
                tail = out_ids[p_len:]
                # Prefer counting by non-PAD tokens; fall back to EOS-based cutoff if no PAD
                if tok.pad_token_id is not None:
                    real_len = int((tail != tok.pad_token_id).sum().item())
                else:
                    # No PAD available: truncate at first EOS if present
                    if tok.eos_token_id is not None:
                        eos_pos = (tail == tok.eos_token_id).nonzero(as_tuple=True)[0]
                        real_len = int(eos_pos[0].item()) if eos_pos.numel() > 0 else int(tail.size(0))
                    else:
                        real_len = int(tail.size(0))

                gen_len_tok = max(0, real_len)
                total_gen_tokens += gen_len_tok

                content_ids = tail[:gen_len_tok]
                text = tok.decode(content_ids, skip_special_tokens=True)
                total_gen_chars += len(text)

                results.append({
                    "idx": len(results),
                    "prompt": chunk[i],
                    "generated": text,
                    "gen_len_tokens": gen_len_tok,
                    "gen_len_chars": len(text),
                })
            if bar is not None:
                bar.update(len(chunk))
        if bar is not None:
            bar.close()

    n = max(1, len(prompts))
    avg_tok = total_gen_tokens / n
    avg_chr = total_gen_chars / n
    print("\n[summary]")
    print(f"Samples: {len(prompts)}")
    print(f"Avg generated length: {avg_tok:.2f} tokens, {avg_chr:.2f} chars")
    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[ok] Saved outputs to: {args.save_jsonl}")


if __name__ == "__main__":
    main()
