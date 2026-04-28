from __future__ import annotations

import re


MCQ_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalize_text(text: str) -> str:
    if text is None:
        text = ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\-]", "", text)
    return text.strip()


def extract_mcq_choice(text: str) -> str | None:
    match = MCQ_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def extract_numeric(text: str) -> str | None:
    matches = NUM_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].lstrip("+")
