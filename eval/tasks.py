from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import os
import torch
from datasets import load_dataset  # type: ignore

from .generation import generate_batched
from .model_loader import load_model_and_tokenizer
from .dist import init_distributed, shard_list, all_reduce_sum_tensor
from .normalize import (
    extract_gsm8k_gold,
    extract_math_gold,
    extract_math_pred,
    normalize_number,
    extract_gsm8k_pred,
    cleanup_repetition,
)
# instruction-following evaluation removed per request; use external AlpacaEval/MT-Bench repos


@dataclass
class EvalConfig:
    model: str
    base_model: Optional[str]
    dtype: str
    max_new_tokens: int
    temperature: float
    top_p: float
    batch_size: int
    n_samples: Optional[int]
    save_outputs: Optional[str]


def _accuracy(pairs: Iterable[Tuple[Optional[str], Optional[str]]]) -> Tuple[float, int, int]:
    total = 0
    correct = 0
    for pred, gold in pairs:
        total += 1
        if pred is not None and gold is not None and str(pred).strip() == str(gold).strip():
            correct += 1
    return (correct / total) if total else 0.0, correct, total


def eval_gsm8k(cfg: EvalConfig) -> Dict:
    is_dist, rank, world, local_rank, device = init_distributed()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))
    questions_all = [row["question"] for row in ds]  # type: ignore
    golds_all = [extract_gsm8k_gold(row["answer"]) for row in ds]  # type: ignore

    # Shard across ranks
    idxs = list(range(len(questions_all)))
    shard_idxs = shard_list(idxs, rank, world)
    questions = [questions_all[i] for i in shard_idxs]
    golds = [golds_all[i] for i in shard_idxs]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype, device=device)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
        show_progress=(rank == 0),
        desc=f"gen({rank}/{world})",
    )
    outputs = [cleanup_repetition(o) for o in outputs]
    preds = [extract_gsm8k_pred(o) for o in outputs]
    # Local stats
    local_total = torch.tensor(len(golds), device=device, dtype=torch.long)
    local_correct = torch.tensor(sum(1 for p, g in zip(preds, golds) if p is not None and g is not None and str(p).strip()==str(g).strip()), device=device, dtype=torch.long)
    # Global reduce
    total = int(all_reduce_sum_tensor(local_total).item())
    correct = int(all_reduce_sum_tensor(local_correct).item())
    acc = (correct / total) if total > 0 else 0.0

    # Save outputs (per-rank) with structured folder:
    # <base_dir>/<tag>/<rank+1>-<world>.jsonl, where tag = modelTag_benchmark
    if cfg.save_outputs:
        # Derive tag
        def short_name(s: str) -> str:
            return s.split("/")[-1].strip()

        task_tag = "gsm8k"
        task_tag += f"_{cfg.max_new_tokens}"
        if cfg.base_model:
            base = short_name(cfg.base_model)
            # derive student tag from path or HF id
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tail = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                stu = tail
            else:
                stu = short_name(m)
            tag = f"{base}_{stu}_{task_tag}"
        else:
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tag = f"{parts[-1]}_{task_tag}"
            else:
                tag = f"{short_name(m)}_base_{task_tag}"

        base_dir = os.path.dirname(cfg.save_outputs) if cfg.save_outputs.endswith(".jsonl") else cfg.save_outputs
        out_dir = os.path.join(base_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{rank+1}-{world}.jsonl")

        with open(out_path, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                    "rank": rank,
                }, ensure_ascii=False) + "\n")
        # Rank-0 summary log
        if rank == 0:
            summary = {
                "task": "gsm8k",
                "model": cfg.model,
                "base_model": cfg.base_model,
                "dtype": cfg.dtype,
                "n_samples": cfg.n_samples,
                "batch_size": cfg.batch_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as lf:
                lf.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return {"task": "gsm8k", "accuracy": acc, "correct": correct, "total": total}


def eval_math(cfg: EvalConfig) -> Dict:
    is_dist, rank, world, local_rank, device = init_distributed()
    ds = load_dataset("hendrycks/competition_math", split="test")
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))
    questions_all = [row["problem"] for row in ds]  # type: ignore
    golds_all = [extract_math_gold(row["solution"]) for row in ds]  # type: ignore

    idxs = list(range(len(questions_all)))
    shard_idxs = shard_list(idxs, rank, world)
    questions = [questions_all[i] for i in shard_idxs]
    golds = [golds_all[i] for i in shard_idxs]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype, device=device)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
        show_progress=(rank == 0),
        desc=f"gen({rank}/{world})",
    )
    outputs = [cleanup_repetition(o) for o in outputs]
    preds = [extract_math_pred(o) for o in outputs]
    # Local
    local_total = torch.tensor(len(golds), device=device, dtype=torch.long)
    local_correct = torch.tensor(sum(1 for p, g in zip(preds, golds) if p is not None and g is not None and str(p).strip()==str(g).strip()), device=device, dtype=torch.long)
    # Global
    total = int(all_reduce_sum_tensor(local_total).item())
    correct = int(all_reduce_sum_tensor(local_correct).item())
    acc = (correct / total) if total > 0 else 0.0

    if cfg.save_outputs:
        def short_name(s: str) -> str:
            return s.split("/")[-1].strip()

        task_tag = "math"
        task_tag += f"_{cfg.max_new_tokens}"
        if cfg.base_model:
            base = short_name(cfg.base_model)
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tail = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                stu = tail
            else:
                stu = short_name(m)
            tag = f"{base}_{stu}_{task_tag}"
        else:
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tag = f"{parts[-1]}_{task_tag}"
            else:
                tag = f"{short_name(m)}_base_{task_tag}"

        base_dir = os.path.dirname(cfg.save_outputs) if cfg.save_outputs.endswith(".jsonl") else cfg.save_outputs
        out_dir = os.path.join(base_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{rank+1}-{world}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                    "rank": rank,
                }, ensure_ascii=False) + "\n")
        if rank == 0:
            summary = {
                "task": "math",
                "model": cfg.model,
                "base_model": cfg.base_model,
                "dtype": cfg.dtype,
                "n_samples": cfg.n_samples,
                "batch_size": cfg.batch_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as lf:
                lf.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return {"task": "math", "accuracy": acc, "correct": correct, "total": total}


def eval_aime_jsonl(cfg: EvalConfig, jsonl_file: str) -> Dict:
    """Temporary AIME evaluator from a local JSONL with fields {question, answer}.

    Gold and predictions use numeric normalization like GSM8K.
    """
    import io
    is_dist, rank, world, local_rank, device = init_distributed()
    questions_all: List[str] = []
    golds_all: List[Optional[str]] = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            questions_all.append(q)
            golds_all.append(normalize_number(a))
    if cfg.n_samples is not None:
        questions_all = questions_all[: cfg.n_samples]
        golds_all = golds_all[: cfg.n_samples]

    idxs = list(range(len(questions_all)))
    shard_idxs = shard_list(idxs, rank, world)
    questions = [questions_all[i] for i in shard_idxs]
    golds = [golds_all[i] for i in shard_idxs]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype, device=device)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
        show_progress=(rank == 0),
        desc=f"gen({rank}/{world})",
    )
    outputs = [cleanup_repetition(o) for o in outputs]
    preds = [normalize_number(o) for o in outputs]
    local_total = torch.tensor(len(golds), device=device, dtype=torch.long)
    local_correct = torch.tensor(sum(1 for p, g in zip(preds, golds) if p is not None and g is not None and str(p).strip()==str(g).strip()), device=device, dtype=torch.long)
    total = int(all_reduce_sum_tensor(local_total).item())
    correct = int(all_reduce_sum_tensor(local_correct).item())
    acc = (correct / total) if total > 0 else 0.0

    if cfg.save_outputs:
        def short_name(s: str) -> str:
            return s.split("/")[-1].strip()

        task_tag = "aime_jsonl"
        task_tag += f"_{cfg.max_new_tokens}"
        if cfg.base_model:
            base = short_name(cfg.base_model)
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tail = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                stu = tail
            else:
                stu = short_name(m)
            tag = f"{base}_{stu}_{task_tag}"
        else:
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tag = f"{parts[-1]}_{task_tag}"
            else:
                tag = f"{short_name(m)}_base_{task_tag}"

        base_dir = os.path.dirname(cfg.save_outputs) if cfg.save_outputs.endswith(".jsonl") else cfg.save_outputs
        out_dir = os.path.join(base_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{rank+1}-{world}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                    "rank": rank,
                }, ensure_ascii=False) + "\n")
        if rank == 0:
            summary = {
                "task": "aime_jsonl",
                "model": cfg.model,
                "base_model": cfg.base_model,
                "dtype": cfg.dtype,
                "n_samples": cfg.n_samples,
                "batch_size": cfg.batch_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "source": jsonl_file,
            }
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as lf:
                lf.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return {"task": "aime_jsonl", "accuracy": acc, "correct": correct, "total": total}


def eval_aime_hf(cfg: EvalConfig, split: str = "test") -> Dict:
    """Evaluate on HuggingFaceH4/aime_2024.

    This function is defensive about field names; it will try common keys for
    question and answer. Answers are normalized numerically for exact match.
    """
    is_dist, rank, world, local_rank, device = init_distributed()
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split)
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))

    def pick(row: Dict, keys: List[str]) -> Optional[str]:
        for k in keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    # Common field candidates
    q_keys = ["question", "problem", "prompt", "input"]
    a_keys = ["answer", "final_answer", "label", "solution", "target"]

    questions_all: List[str] = []
    golds_all: List[Optional[str]] = []
    for row in ds:  # type: ignore
        q = pick(row, q_keys) or ""
        a_raw = pick(row, a_keys) or ""
        questions_all.append(q)
        golds_all.append(normalize_number(a_raw))

    idxs = list(range(len(questions_all)))
    shard_idxs = shard_list(idxs, rank, world)
    questions = [questions_all[i] for i in shard_idxs]
    golds = [golds_all[i] for i in shard_idxs]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype, device=device)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
        show_progress=(rank == 0),
        desc=f"gen({rank}/{world})",
    )
    preds = [normalize_number(o) for o in outputs]
    local_total = torch.tensor(len(golds), device=device, dtype=torch.long)
    local_correct = torch.tensor(sum(1 for p, g in zip(preds, golds) if p is not None and g is not None and str(p).strip()==str(g).strip()), device=device, dtype=torch.long)
    total = int(all_reduce_sum_tensor(local_total).item())
    correct = int(all_reduce_sum_tensor(local_correct).item())
    acc = (correct / total) if total > 0 else 0.0

    if cfg.save_outputs:
        def short_name(s: str) -> str:
            return s.split("/")[-1].strip()

        task_tag = f"aime_2024_{split}"
        task_tag += f"_{cfg.max_new_tokens}"
        if cfg.base_model:
            base = short_name(cfg.base_model)
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tail = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                stu = tail
            else:
                stu = short_name(m)
            tag = f"{base}_{stu}_{task_tag}"
        else:
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tag = f"{parts[-1]}_{task_tag}"
            else:
                tag = f"{short_name(m)}_base_{task_tag}"

        base_dir = os.path.dirname(cfg.save_outputs) if cfg.save_outputs.endswith(".jsonl") else cfg.save_outputs
        out_dir = os.path.join(base_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{rank+1}-{world}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                    "rank": rank,
                }, ensure_ascii=False) + "\n")
        if rank == 0:
            summary = {
                "task": f"aime_2024:{split}",
                "model": cfg.model,
                "base_model": cfg.base_model,
                "dtype": cfg.dtype,
                "n_samples": cfg.n_samples,
                "batch_size": cfg.batch_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as lf:
                lf.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return {"task": f"aime_2024:{split}", "accuracy": acc, "correct": correct, "total": total}


def eval_math500_hf(cfg: EvalConfig, split: str = "test") -> Dict:
    """Evaluate on HuggingFaceH4/MATH-500.

    Tries to parse answers from solution (\\boxed{...}) or numeric fields.
    """
    is_dist, rank, world, local_rank, device = init_distributed()
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))

    def pick(row: Dict, keys: List[str]) -> Optional[str]:
        for k in keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    q_keys = ["problem", "question", "prompt", "input"]
    sol_keys = ["solution", "rationale", "explanation"]
    ans_keys = ["final_answer", "answer", "label", "target"]

    questions_all: List[str] = []
    golds_all: List[Optional[str]] = []
    for row in ds:  # type: ignore
        q = pick(row, q_keys) or ""
        sol = pick(row, sol_keys) or ""
        fa = pick(row, ans_keys) or ""
        g = extract_math_gold(sol) if sol else None
        if g is None and fa:
            g = normalize_number(fa)
        questions_all.append(q)
        golds_all.append(g)

    idxs = list(range(len(questions_all)))
    shard_idxs = shard_list(idxs, rank, world)
    questions = [questions_all[i] for i in shard_idxs]
    golds = [golds_all[i] for i in shard_idxs]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype, device=device)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
        show_progress=(rank == 0),
        desc=f"gen({rank}/{world})",
    )
    outputs = [cleanup_repetition(o) for o in outputs]
    preds = [extract_math_pred(o) for o in outputs]
    local_total = torch.tensor(len(golds), device=device, dtype=torch.long)
    local_correct = torch.tensor(sum(1 for p, g in zip(preds, golds) if p is not None and g is not None and str(p).strip()==str(g).strip()), device=device, dtype=torch.long)
    total = int(all_reduce_sum_tensor(local_total).item())
    correct = int(all_reduce_sum_tensor(local_correct).item())
    acc = (correct / total) if total > 0 else 0.0

    if cfg.save_outputs:
        def short_name(s: str) -> str:
            return s.split("/")[-1].strip()

        task_tag = f"MATH-500_{split}"
        task_tag += f"_{cfg.max_new_tokens}"
        if cfg.base_model:
            base = short_name(cfg.base_model)
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tail = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                stu = tail
            else:
                stu = short_name(m)
            tag = f"{base}_{stu}_{task_tag}"
        else:
            m = cfg.model
            if os.path.sep in m:
                parts = [p for p in os.path.normpath(m).split(os.path.sep) if p]
                tag = f"{parts[-1]}_{task_tag}"
            else:
                tag = f"{short_name(m)}_base_{task_tag}"

        base_dir = os.path.dirname(cfg.save_outputs) if cfg.save_outputs.endswith(".jsonl") else cfg.save_outputs
        out_dir = os.path.join(base_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{rank+1}-{world}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                    "rank": rank,
                }, ensure_ascii=False) + "\n")
        if rank == 0:
            summary = {
                "task": f"MATH-500:{split}",
                "model": cfg.model,
                "base_model": cfg.base_model,
                "dtype": cfg.dtype,
                "n_samples": cfg.n_samples,
                "batch_size": cfg.batch_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as lf:
                lf.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return {"task": f"MATH-500:{split}", "accuracy": acc, "correct": correct, "total": total}


# Instruction-following evaluation intentionally removed; please use
# the official/open-source AlpacaEval or MT-Bench tooling externally.
