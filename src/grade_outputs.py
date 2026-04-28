from __future__ import annotations

import pandas as pd

from .normalize_outputs import extract_mcq_choice, extract_numeric, normalize_text


def grade_row(row: pd.Series) -> str:
    if not row.get("success", False):
        return "wrong"

    grading_type = (row.get("grading_type") or "unknown").lower()
    expected = row.get("expected_answer")
    response = row.get("response_text") or ""
    if expected is None or str(expected).strip() == "":
        return "unknown"
    expected_str = str(expected).strip()

    if grading_type == "mcq":
        pred = extract_mcq_choice(response)
        return "correct" if pred and pred == expected_str.upper() else "wrong"

    if grading_type == "numeric":
        pred = extract_numeric(response)
        expected_num = extract_numeric(expected_str)
        if not pred or not expected_num:
            return "unknown"
        return "correct" if pred == expected_num else "wrong"

    if grading_type == "exact":
        return "correct" if normalize_text(response) == normalize_text(expected_str) else "wrong"

    if grading_type == "contains":
        return "correct" if normalize_text(expected_str) in normalize_text(response) else "wrong"

    return "unknown"


def grade_outputs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    graded = df.copy()
    graded["correctness"] = graded.apply(grade_row, axis=1)
    return graded
