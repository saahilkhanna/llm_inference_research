from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def _bucket(vllm: str, sglang: str) -> str:
    if vllm == "correct" and sglang == "correct":
        return "both_correct"
    if vllm == "wrong" and sglang == "wrong":
        return "both_wrong"
    if vllm == "correct" and sglang == "wrong":
        return "vllm_correct_sglang_wrong"
    if vllm == "wrong" and sglang == "correct":
        return "sglang_correct_vllm_wrong"
    if vllm == "unknown" and sglang == "unknown":
        return "both_unknown"
    return "mixed_unknown"


def _compare_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "optimization_mode" not in df.columns:
        df["optimization_mode"] = "baseline"
    if "repeat_index" not in df.columns:
        df["repeat_index"] = 1
    pivot = df.pivot_table(
        index=[
            "sample_id",
            "workload",
            "task_id",
            "workload_class",
            "optimization_mode",
            "repeat_index",
            "prompt",
            "expected_answer",
            "grading_type",
        ],
        columns="backend",
        values=["response_text", "correctness", "latency_seconds", "success", "error_message"],
        aggfunc="first",
    ).reset_index()
    pivot.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in pivot.columns]

    pivot["bucket"] = pivot.apply(
        lambda row: _bucket(
            row.get("correctness_vllm", "unknown"),
            row.get("correctness_sglang", "unknown"),
        ),
        axis=1,
    )
    return pivot


def compare_backends(graded_df: pd.DataFrame, run_dir: Path) -> dict[str, Path]:
    ensure_dir(run_dir / "processed")
    public_path = run_dir / "processed" / "public_per_sample_comparison.csv"
    custom_path = run_dir / "processed" / "custom_per_sample_comparison.csv"
    summary_path = run_dir / "processed" / "failure_bucket_summary.csv"
    engine_summary_path = run_dir / "processed" / "engine_mode_summary.csv"

    if graded_df.empty:
        pd.DataFrame().to_csv(public_path, index=False)
        pd.DataFrame().to_csv(custom_path, index=False)
        pd.DataFrame(columns=["bucket", "count"]).to_csv(summary_path, index=False)
        pd.DataFrame(
            columns=["backend", "optimization_mode", "repeat_index", "workload", "correct_rate", "wrong_rate", "unknown_rate"]
        ).to_csv(engine_summary_path, index=False)
        return {
            "public_comparison": public_path,
            "custom_comparison": custom_path,
            "bucket_summary": summary_path,
            "engine_mode_summary": engine_summary_path,
        }

    # Comparison tables include every configured engine. The primary failure bucket
    # remains vLLM vs SGLang so older reports stay comparable.
    public_df = graded_df[graded_df["workload"] == "public"].copy()
    custom_df = graded_df[graded_df["workload"] == "custom"].copy()

    public_cmp = _compare_subset(public_df)
    custom_cmp = _compare_subset(custom_df)

    public_cmp.to_csv(public_path, index=False)
    custom_cmp.to_csv(custom_path, index=False)

    bucket_parts: list[pd.DataFrame] = []
    if "bucket" in public_cmp.columns:
        bucket_parts.append(public_cmp[["bucket"]])
    if "bucket" in custom_cmp.columns:
        bucket_parts.append(custom_cmp[["bucket"]])
    bucket_df = pd.concat(bucket_parts, ignore_index=True) if bucket_parts else pd.DataFrame(columns=["bucket"])
    if bucket_df.empty:
        summary = pd.DataFrame(columns=["bucket", "count"])
    else:
        summary = bucket_df.value_counts("bucket").reset_index(name="count")
    summary.to_csv(summary_path, index=False)

    # Additional per-engine summary across all configured engines/modes/repeats.
    df = graded_df.copy()
    if "optimization_mode" not in df.columns:
        df["optimization_mode"] = "baseline"
    if "repeat_index" not in df.columns:
        df["repeat_index"] = 1
    rows = []
    for (backend, mode, rep, workload), grp in df.groupby(["backend", "optimization_mode", "repeat_index", "workload"]):
        rows.append(
            {
                "backend": backend,
                "optimization_mode": mode,
                "repeat_index": int(rep),
                "workload": workload,
                "correct_rate": float((grp["correctness"] == "correct").mean()),
                "wrong_rate": float((grp["correctness"] == "wrong").mean()),
                "unknown_rate": float((grp["correctness"] == "unknown").mean()),
            }
        )
    pd.DataFrame(rows).to_csv(engine_summary_path, index=False)

    return {
        "public_comparison": public_path,
        "custom_comparison": custom_path,
        "bucket_summary": summary_path,
        "engine_mode_summary": engine_summary_path,
    }
