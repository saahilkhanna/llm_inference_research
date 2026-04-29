from __future__ import annotations

import pandas as pd

from .normalize_outputs import extract_mcq_choice, extract_numeric, normalize_text


def is_benchmark_valid_for_claims(row: pd.Series) -> bool:
    grading_type = (row.get("grading_type") or "unknown").lower()
    task_id = str(row.get("task_id") or "").lower()
    if grading_type in {"numeric", "mcq", "exact"}:
        return True
    # HumanEval prompt-level contains checks are diagnostic only; benchmark-valid
    # coding claims must come from standard evaluator test execution.
    if task_id.startswith("humaneval"):
        return False
    return False


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

    if grading_type == "contains_diagnostic":
        return "unknown"

    return "unknown"


def grade_outputs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    graded = df.copy()
    graded["correctness"] = graded.apply(grade_row, axis=1)
    graded["benchmark_valid_for_claims"] = graded.apply(is_benchmark_valid_for_claims, axis=1)
    graded["claim_exclusion_reason"] = ""
    graded.loc[~graded["benchmark_valid_for_claims"], "claim_exclusion_reason"] = "diagnostic_or_unscored"
    graded["missing_for_claims"] = ~graded["benchmark_valid_for_claims"]

    def _failure_tag(row: pd.Series) -> str:
        if bool(row.get("success", False)) is False:
            msg = str(row.get("error_message", "")).lower()
            if "timeout" in msg:
                return "runtime_timeout"
            if "429" in msg or "rate limit" in msg:
                return "runtime_rate_limit"
            if "connection" in msg or "ssl" in msg:
                return "runtime_connection"
            return "runtime_error"
        correctness = str(row.get("correctness", "unknown")).lower()
        if correctness == "correct":
            return "correct"
        if correctness == "wrong":
            if str(row.get("response_text", "")).strip() == "":
                return "empty_response"
            if str(row.get("grading_type", "")).lower() == "numeric" and row.get("response_numeric_extracted") in {None, ""}:
                return "numeric_extraction_miss"
            return "semantic_mismatch"
        if str(row.get("grading_type", "")).lower() == "contains_diagnostic":
            return "diagnostic_only"
        return "unknown_ungraded"

    graded["failure_tag"] = graded.apply(_failure_tag, axis=1)
    return graded
