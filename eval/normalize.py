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


def extract_gsm8k_pred(text: str) -> Optional[str]:
    """Heuristic extractor for GSM8K model outputs.

    Preference order:
    1) Last \boxed{...}
    2) Number in the last line containing keywords like 'Answer', 'final', '='
    3) Fallback to last standalone number
    """
    if not text:
        return None
    # 1) Prefer boxed
    boxes = _BOX_RE.findall(text)
    if boxes:
        cand = boxes[-1].strip().strip("$")
        num = normalize_number(cand)
        if num is not None:
            return num
    # 2) Look at lines near the end with keywords
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    keywords = ("answer", "final", "=", ":")
    for ln in reversed(lines[-10:]):
        low = ln.lower()
        if any(k in low for k in keywords):
            num = normalize_number(ln)
            if num is not None:
                return num
    # 3) Fallback last number
    return normalize_number(text)


def cleanup_repetition(text: str, max_repeat: int = 3) -> str:
    """Collapse excessive trailing repeated lines (e.g., 'Answer: 6')."""
    lines = text.splitlines()
    if not lines:
        return text
    # Count repeats of last non-empty line
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return text
    tail = lines[i].strip()
    cnt = 0
    j = i
    while j >= 0 and lines[j].strip() == tail:
        cnt += 1
        j -= 1
    if cnt > max_repeat:
        kept = lines[: j + 1] + [tail] * max_repeat + lines[i + 1 :]
        return "\n".join(kept)
    return text


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
