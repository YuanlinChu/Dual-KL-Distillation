"""
Check OpenThoughts3 token lengths and truncation impact under the local SFT pipeline.

Example:
  python tinker_cookbook/scripts/check_openthoughts_lengths.py \
    --dataset open-thoughts/OpenThoughts3-1.2M \
    --split train \
    --model-name Qwen/Qwen3-8B-Base \
    --renderer-name qwen3 \
    --max-length 4096 \
    --num-samples 1000
"""

from __future__ import annotations

import argparse
import math
from statistics import mean
from typing import Iterable

import datasets
from transformers import AutoTokenizer

from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights


def _messages_from_row(row: dict) -> list[renderers.Message]:
    conversations = row.get("conversations", [])
    return [
        {
            "role": "user" if msg["from"] == "human" else "assistant",
            "content": msg["value"],
        }
        for msg in conversations
    ]


def _percentile(values: list[int | float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return float(values[0])
    if pct >= 100:
        return float(values[-1])
    idx = int(math.ceil((pct / 100.0) * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def _summarize(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    sorted_vals = sorted(values)
    return {
        "count": float(len(values)),
        "min": float(sorted_vals[0]),
        "p50": _percentile(sorted_vals, 50),
        "p90": _percentile(sorted_vals, 90),
        "p99": _percentile(sorted_vals, 99),
        "max": float(sorted_vals[-1]),
        "mean": float(mean(values)),
    }


def _print_stats(title: str, stats: dict[str, float]) -> None:
    print(f"{title}:")
    for key in ("count", "min", "p50", "p90", "p99", "max", "mean"):
        if key in stats:
            print(f"  {key}: {stats[key]:.2f}")


def iter_rows(ds: Iterable[dict], n: int) -> Iterable[dict]:
    count = 0
    for row in ds:
        if count >= n:
            break
        yield row
        count += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="open-thoughts/OpenThoughts3-1.2M")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--renderer-name", default="qwen3")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    ds = datasets.load_dataset(args.dataset, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    full_lengths: list[int] = []
    trunc_lengths: list[int] = []
    full_asst_tokens: list[int] = []
    trunc_asst_tokens: list[int] = []
    dropped_frac: list[float] = []
    trunc_count = 0

    for row in iter_rows(ds, args.num_samples):
        messages = _messages_from_row(row)
        model_input, weights = renderer.build_supervised_example(
            messages, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        full_len = sum(chunk.length for chunk in model_input.chunks)
        full_asst = int(weights.sum().item())

        datum = datum_from_model_input_weights(model_input, weights, max_length=args.max_length)
        trunc_len = len(datum.loss_fn_inputs["target_tokens"].data) + 1
        trunc_asst = int(sum(datum.loss_fn_inputs["weights"].data))

        full_lengths.append(full_len)
        trunc_lengths.append(trunc_len)
        full_asst_tokens.append(full_asst)
        trunc_asst_tokens.append(trunc_asst)

        if full_len > args.max_length:
            trunc_count += 1
            if full_asst > 0:
                dropped_frac.append((full_asst - trunc_asst) / full_asst)

    print(f"samples: {len(full_lengths)}")
    print(f"truncated: {trunc_count} ({trunc_count / max(len(full_lengths), 1):.2%})")
    _print_stats("full_total_tokens", _summarize(full_lengths))
    _print_stats("trunc_total_tokens", _summarize(trunc_lengths))
    _print_stats("full_assistant_tokens", _summarize(full_asst_tokens))
    _print_stats("trunc_assistant_tokens", _summarize(trunc_asst_tokens))
    if dropped_frac:
        _print_stats("assistant_tokens_dropped_frac", _summarize(dropped_frac))


if __name__ == "__main__":
    main()
