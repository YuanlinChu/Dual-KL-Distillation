#!/usr/bin/env python3
"""
Sample a subset from a Hugging Face dataset (e.g., DeepMath) and optionally
push it to the Hub as a new dataset repository.

Example:
  python scripts/sample_deepmath.py \
    --source <source-dataset-id> \
    --split train \
    --num-samples 32000 \
    --seed 42 \
    --repo-id zwhe99/DeepMath-103K \
    --push-to-hub

- 采样并推送（示例）:
      - python sample_deepmath.py --source zwhe99/DeepMath-103K --split train --num-samples 32000 --seed 42 --repo-id zwhe99/DeepMath-
        103K --push-to-hub
- 可选本地保存:
      - python sample_deepmath.py --source zwhe99/DeepMath-103K --split train --num-samples 32000 --output-dir DeepMath-32k

Notes:
- You must have `datasets` installed: pip install datasets
- To push to hub, set `HF_TOKEN` env var or be logged in via `huggingface-cli login`.
"""

import argparse
import os
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict


def sample_dataset(
    source: str,
    split: str,
    num_samples: int,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset(source, split=split)

    # Ensure we don't request more than available
    n = min(num_samples, len(ds))
    if n < num_samples:
        print(f"[warn] Requested {num_samples} but split has only {len(ds)} rows. Sampling {n}.")

    # Shuffle for uniform random sampling, then take first n
    ds_shuffled = ds.shuffle(seed=seed)
    sampled = ds_shuffled.select(range(n))
    return sampled


def maybe_save_local(ds: Dataset, output_dir: Optional[str]) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    # Save as a DatasetDict with a single split for easier reuse
    dd = DatasetDict({"train": ds})
    dd.save_to_disk(output_dir)
    print(f"[ok] Saved sampled dataset to: {output_dir}")


def maybe_push_to_hub(ds: Dataset, repo_id: Optional[str], private: bool, commit_message: str) -> None:
    if not repo_id:
        return
    token_present = bool(os.environ.get("HF_TOKEN"))
    if not token_present:
        print("[warn] HF_TOKEN not set. Ensure you're logged in via `huggingface-cli login` or set HF_TOKEN.")

    # Push as a single-split dataset
    print(f"[info] Pushing dataset to Hub: {repo_id} (private={private})")
    ds.push_to_hub(repo_id=repo_id, private=private, commit_message=commit_message)
    print("[ok] Push completed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Randomly sample a subset from a HF dataset and optionally push to Hub.")
    p.add_argument("--source", required=True, help="Source dataset ID or local path for `datasets.load_dataset`, e.g. 'deepmind/mathematics_dataset' or 'deepmath'.")
    p.add_argument("--split", default="train", help="Split to sample from (default: train).")
    p.add_argument("--num-samples", type=int, default=32000, help="Number of samples to draw (default: 32000).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling (default: 42).")

    p.add_argument("--output-dir", default=None, help="Optional local directory to save the sampled dataset via save_to_disk.")

    p.add_argument("--push-to-hub", action="store_true", help="Push the sampled dataset to Hugging Face Hub.")
    p.add_argument("--repo-id", default=None, help="Target Hub repo id, e.g. 'zwhe99/DeepMath-103K'. Required if --push-to-hub.")
    p.add_argument("--private", action="store_true", help="Create/push as a private dataset on the Hub.")
    p.add_argument("--commit-message", default="Add sampled subset", help="Commit message for Hub push.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.push_to_hub and not args.repo_id:
        raise SystemExit("--repo-id is required when using --push-to-hub")

    print(f"[info] Loading '{args.source}' split='{args.split}' ...")
    sampled = sample_dataset(
        source=args.source,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f"[ok] Sampled {len(sampled)} rows.")

    maybe_save_local(sampled, args.output_dir)

    if args.push_to_hub:
        maybe_push_to_hub(
            ds=sampled,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()

