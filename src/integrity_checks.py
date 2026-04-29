from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .normalize_outputs import extract_numeric, normalize_text
from .utils import append_jsonl, write_json


def _expected_eval_conditions(
    config: AppConfig, conditions: list[tuple[str, int, str]]
) -> set[tuple[str, str, int]]:
    expected: set[tuple[str, str, int]] = set()
    for optimization_mode, repeat_index, backend in conditions:
        if not config.standard_eval_enabled or config.standard_evaluator == "none":
            continue
        if config.standard_eval_run_on_all_conditions:
            expected.add((backend, optimization_mode, repeat_index))
        elif optimization_mode == "baseline" and repeat_index == 1:
            expected.add((backend, optimization_mode, repeat_index))
    return expected


def validate_pre_run_integrity(
    config: AppConfig, run_dir: Path, samples: list[dict[str, Any]], smoke: bool
) -> None:
    smoke_like_run = smoke or ("smoke" in (config.run_name or "").lower())
    if not smoke_like_run and config.limit <= 10:
        raise ValueError(
            f"Non-smoke run LIMIT={config.limit} is too small and risks muddied conclusions. "
            "Use >10 for non-smoke experiments."
        )
    if not smoke_like_run and 0 < config.standard_eval_limit_override <= 10:
        raise ValueError(
            f"Non-smoke STANDARD_EVAL_LIMIT_OVERRIDE={config.standard_eval_limit_override} is too small. "
            "Use >10 or disable override."
        )

    sample_ids = [str(s.get("sample_id", "")).strip() for s in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values detected; sample alignment would be invalid.")

    gradeable = {"numeric", "mcq", "exact", "contains"}
    missing_expected: list[str] = []
    prompt_leaks: list[str] = []
    gsm8k_expected_not_numeric: list[str] = []

    for sample in samples:
        sample_id = str(sample.get("sample_id", "unknown"))
        grading_type = str(sample.get("grading_type", "unknown")).lower()
        expected = sample.get("expected_answer")
        prompt = str(sample.get("prompt", "") or "")
        task_id = str(sample.get("task_id", ""))
        if grading_type in gradeable and (expected is None or str(expected).strip() == ""):
            missing_expected.append(sample_id)
            continue
        if expected is not None and str(expected).strip():
            expected_norm = normalize_text(str(expected))
            prompt_norm = normalize_text(prompt)
            # "contains" grading often uses identifiers (e.g., function names) that intentionally
            # appear in prompts; leakage checks focus on strict answer-comparison modes.
            if grading_type in {"numeric", "mcq", "exact"} and expected_norm:
                leakage_cues = [
                    f"answer: {expected_norm}",
                    f"final answer: {expected_norm}",
                    f"correct answer: {expected_norm}",
                    f"gold answer: {expected_norm}",
                    f"expected answer: {expected_norm}",
                ]
                # Use cue-based matching to avoid false positives where the expected token
                # legitimately appears in long-context prompts.
                if any(cue in prompt_norm for cue in leakage_cues):
                    prompt_leaks.append(sample_id)
                elif grading_type == "mcq":
                    # MCQ labels can be short; only treat as leakage when paired with explicit cues.
                    patt = rf"(answer|correct answer|final answer)\s*[:=]\s*{re.escape(expected_norm)}\b"
                    if re.search(patt, prompt_norm):
                        prompt_leaks.append(sample_id)
            if task_id.startswith("gsm8k") and grading_type == "numeric":
                if extract_numeric(str(expected)) is None:
                    gsm8k_expected_not_numeric.append(sample_id)

    if missing_expected:
        raise ValueError(
            f"Missing expected_answer for gradeable samples: {missing_expected[:10]} "
            f"(total={len(missing_expected)})"
        )
    if prompt_leaks:
        raise ValueError(
            f"Potential prompt leakage detected (expected answer appears in prompt) for sample_ids: "
            f"{prompt_leaks[:10]} (total={len(prompt_leaks)})"
        )
    if gsm8k_expected_not_numeric:
        raise ValueError(
            f"GSM8K expected answers are not parseable as numeric for sample_ids: "
            f"{gsm8k_expected_not_numeric[:10]} (total={len(gsm8k_expected_not_numeric)})"
        )

    write_json(
        run_dir / "logs" / "integrity_precheck.json",
        {
            "sample_count": len(samples),
            "unique_sample_ids": len(set(sample_ids)),
            "smoke_like_run": smoke_like_run,
            "limit": config.limit,
            "standard_eval_limit_override": config.standard_eval_limit_override,
            "status": "passed",
        },
    )


def validate_post_run_integrity(
    config: AppConfig,
    run_dir: Path,
    graded: pd.DataFrame,
    conditions: list[tuple[str, int, str]],
) -> None:
    logs_path = run_dir / "logs" / "integrity_checks.jsonl"
    if graded.empty:
        raise ValueError("Parsed/graded table is empty; no valid evaluation data collected.")

    expected_sample_ids = set(str(v) for v in graded["sample_id"].dropna().unique().tolist())
    condition_groups = graded.groupby(["backend", "optimization_mode", "repeat_index"])
    for backend, mode, rep in {(e, m, r) for m, r, e in conditions}:
        key_rows = condition_groups.get_group((backend, mode, rep)) if (backend, mode, rep) in condition_groups.groups else None
        if key_rows is None or key_rows.empty:
            raise ValueError(f"Missing graded rows for condition {backend}:{mode}:repeat_{rep}")
        sample_ids = set(str(v) for v in key_rows["sample_id"].dropna().tolist())
        if sample_ids != expected_sample_ids:
            raise ValueError(
                f"Sample ID mismatch for condition {backend}:{mode}:repeat_{rep}; "
                f"expected {len(expected_sample_ids)} ids, got {len(sample_ids)} ids."
            )

    if "correctness" not in graded.columns:
        raise ValueError("No correctness column in graded outputs; gold comparison not applied.")
    if not ((graded["correctness"] == "correct") | (graded["correctness"] == "wrong")).any():
        raise ValueError("All graded rows are 'unknown'; gold comparison appears ineffective.")

    gsm8k_rows = graded[graded["task_id"].astype(str).str.startswith("gsm8k") & (graded["grading_type"] == "numeric")]
    if not gsm8k_rows.empty:
        pred_missing = gsm8k_rows["response_text"].astype(str).map(extract_numeric).isna().sum()
        append_jsonl(
            logs_path,
            {
                "event": "gsm8k_numeric_extraction_check",
                "rows": int(len(gsm8k_rows)),
                "missing_numeric_prediction": int(pred_missing),
                "missing_rate": float(pred_missing / len(gsm8k_rows)),
            },
        )
        if pred_missing == len(gsm8k_rows):
            raise ValueError("GSM8K numeric extraction failed for all rows.")

    if config.standard_eval_enabled and config.standard_evaluator != "none":
        expected_conditions = _expected_eval_conditions(config, conditions)
        standard_log_path = run_dir / "logs" / "standard_evaluator.jsonl"
        if not standard_log_path.exists():
            raise ValueError("STANDARD_EVAL_ENABLED=true but standard_evaluator log is missing.")
        records = []
        for line in standard_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        completed = {
            (
                r.get("backend"),
                r.get("optimization_mode", "baseline"),
                int(r.get("repeat_index", 1)),
            )
            for r in records
            if r.get("event") == "lm_eval_completed"
        }
        failed = {
            (
                r.get("backend"),
                r.get("optimization_mode", "baseline"),
                int(r.get("repeat_index", 1)),
            )
            for r in records
            if r.get("event") == "lm_eval_failed"
        }
        missing = expected_conditions - completed
        if failed:
            raise ValueError(f"lm_eval failed for conditions: {sorted(failed)}")
        if missing:
            raise ValueError(f"lm_eval did not complete for conditions: {sorted(missing)}")

    append_jsonl(logs_path, {"event": "integrity_postcheck_passed", "rows": int(len(graded))})
