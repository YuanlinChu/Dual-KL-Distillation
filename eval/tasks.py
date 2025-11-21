from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset  # type: ignore

from .generation import generate_batched
from .model_loader import load_model_and_tokenizer
from .normalize import (
    extract_gsm8k_gold,
    extract_math_gold,
    extract_math_pred,
    normalize_number,
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
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))
    questions = [row["question"] for row in ds]  # type: ignore
    golds = [extract_gsm8k_gold(row["answer"]) for row in ds]  # type: ignore

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
    )
    preds = [extract_gsm8k_gold(o) for o in outputs]
    acc, correct, total = _accuracy(zip(preds, golds))

    if cfg.save_outputs:
        with open(cfg.save_outputs, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                }, ensure_ascii=False) + "\n")

    return {"task": "gsm8k", "accuracy": acc, "correct": correct, "total": total}


def eval_math(cfg: EvalConfig) -> Dict:
    ds = load_dataset("hendrycks/competition_math", split="test")
    if cfg.n_samples is not None:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))
    questions = [row["problem"] for row in ds]  # type: ignore
    golds = [extract_math_gold(row["solution"]) for row in ds]  # type: ignore

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
    )
    preds = [extract_math_pred(o) for o in outputs]
    acc, correct, total = _accuracy(zip(preds, golds))

    if cfg.save_outputs:
        with open(cfg.save_outputs, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                }, ensure_ascii=False) + "\n")

    return {"task": "math", "accuracy": acc, "correct": correct, "total": total}


def eval_aime_jsonl(cfg: EvalConfig, jsonl_file: str) -> Dict:
    """Temporary AIME evaluator from a local JSONL with fields {question, answer}.

    Gold and predictions use numeric normalization like GSM8K.
    """
    import io
    questions: List[str] = []
    golds: List[Optional[str]] = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            questions.append(q)
            golds.append(normalize_number(a))
    if cfg.n_samples is not None:
        questions = questions[: cfg.n_samples]
        golds = golds[: cfg.n_samples]

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
    )
    from .normalize import normalize_number
    preds = [normalize_number(o) for o in outputs]
    acc, correct, total = _accuracy(zip(preds, golds))

    if cfg.save_outputs:
        with open(cfg.save_outputs, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                }, ensure_ascii=False) + "\n")

    return {"task": "aime_jsonl", "accuracy": acc, "correct": correct, "total": total}


def eval_aime_hf(cfg: EvalConfig, split: str = "test") -> Dict:
    """Evaluate on HuggingFaceH4/aime_2024.

    This function is defensive about field names; it will try common keys for
    question and answer. Answers are normalized numerically for exact match.
    """
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

    questions: List[str] = []
    golds: List[Optional[str]] = []
    for row in ds:  # type: ignore
        q = pick(row, q_keys) or ""
        a_raw = pick(row, a_keys) or ""
        questions.append(q)
        golds.append(normalize_number(a_raw))

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
    )
    preds = [normalize_number(o) for o in outputs]
    acc, correct, total = _accuracy(zip(preds, golds))

    if cfg.save_outputs:
        with open(cfg.save_outputs, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                }, ensure_ascii=False) + "\n")

    return {"task": f"aime_2024:{split}", "accuracy": acc, "correct": correct, "total": total}


def eval_math500_hf(cfg: EvalConfig, split: str = "test") -> Dict:
    """Evaluate on HuggingFaceH4/MATH-500.

    Tries to parse answers from solution (\\boxed{...}) or numeric fields.
    """
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

    questions: List[str] = []
    golds: List[Optional[str]] = []
    for row in ds:  # type: ignore
        q = pick(row, q_keys) or ""
        sol = pick(row, sol_keys) or ""
        fa = pick(row, ans_keys) or ""
        # Prefer solution boxed parsing; fallback to numeric in final answer
        g = extract_math_gold(sol) if sol else None
        if g is None and fa:
            g = normalize_number(fa)
        questions.append(q)
        golds.append(g)

    model, tok = load_model_and_tokenizer(cfg.model, cfg.base_model, cfg.dtype)
    outputs = generate_batched(
        model,
        tok,
        questions,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        batch_size=cfg.batch_size,
    )
    preds = [extract_math_pred(o) for o in outputs]
    acc, correct, total = _accuracy(zip(preds, golds))

    if cfg.save_outputs:
        with open(cfg.save_outputs, "w", encoding="utf-8") as f:
            for q, o, p, g in zip(questions, outputs, preds, golds):
                f.write(json.dumps({
                    "question": q,
                    "output": o,
                    "pred": p,
                    "gold": g,
                    "correct": (p is not None and g is not None and str(p).strip()==str(g).strip()),
                }, ensure_ascii=False) + "\n")

    return {"task": f"MATH-500:{split}", "accuracy": acc, "correct": correct, "total": total}


# Instruction-following evaluation intentionally removed; please use
# the official/open-source AlpacaEval or MT-Bench tooling externally.
