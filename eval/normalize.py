from __future__ import annotations

import re
from typing import Optional


_NUM_RE = re.compile(r"[-+]?\d+(?:[,\d]*\d)?(?:\.\d+)?")
_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _strip(s: str) -> str:
    return s.strip().rstrip(". ")


def normalize_number(text: str) -> Optional[str]:
    """Extract and normalize the last number-like token in text."""
    if not text:
        return None
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    val = matches[-1].group(0)
    val = val.replace(",", "")
    return _strip(val)


def extract_gsm8k_gold(answer: str) -> Optional[str]:
    # Gold often ends with '#### 1234'
    if "####" in answer:
        part = answer.split("####")[-1]
        num = normalize_number(part)
        if num is not None:
            return num
    return normalize_number(answer)


def extract_math_gold(solution: str) -> Optional[str]:
    # Prefer last \boxed{...}
    boxes = _BOX_RE.findall(solution)
    if boxes:
        # Take last box; strip $ and spaces
        cand = boxes[-1].strip().strip("$")
        # If it's purely numeric, normalize; else return raw for exact compare
        num = normalize_number(cand)
        return num if num is not None else _strip(cand)
    # fallback to last number in solution
    return normalize_number(solution)


def extract_math_pred(text: str) -> Optional[str]:
    # Look for \boxed{...} in model output first
    boxes = _BOX_RE.findall(text)
    if boxes:
        cand = boxes[-1].strip().strip("$")
        num = normalize_number(cand)
        return num if num is not None else _strip(cand)
    return normalize_number(text)

