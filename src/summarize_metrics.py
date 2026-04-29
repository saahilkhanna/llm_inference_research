from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def summarize_latency(graded_df: pd.DataFrame, run_dir: Path) -> Path:
    ensure_dir(run_dir / "processed")
    output_path = run_dir / "processed" / "latency_summary.csv"
    if graded_df.empty:
        pd.DataFrame(
            columns=[
                "backend",
                "optimization_mode",
                "repeat_index",
                "workload",
                "workload_class",
                "mean_latency_s",
                "median_latency_s",
                "p95_latency_s",
                "p99_latency_s",
                "success_rate",
                "token_throughput_est",
            ]
        ).to_csv(output_path, index=False)
        return output_path

    rows = []
    df = graded_df.copy()
    if "optimization_mode" not in df.columns:
        df["optimization_mode"] = "baseline"
    if "repeat_index" not in df.columns:
        df["repeat_index"] = 1
    grouped = df.groupby(["backend", "optimization_mode", "repeat_index", "workload", "workload_class"], dropna=False)
    for (backend, optimization_mode, repeat_index, workload, workload_class), grp in grouped:
        latency = grp["latency_seconds"].astype(float)
        token_counts = grp["token_counts"].apply(lambda x: x if isinstance(x, dict) else {})
        total_tokens = token_counts.apply(lambda d: d.get("completion_tokens", 0) or d.get("output_tokens", 0)).sum()
        total_time = latency.sum()
        throughput = float(total_tokens) / total_time if total_time > 0 else 0.0
        rows.append(
            {
                "backend": backend,
                "optimization_mode": optimization_mode,
                "repeat_index": int(repeat_index),
                "workload": workload,
                "workload_class": workload_class,
                "mean_latency_s": float(latency.mean()),
                "median_latency_s": float(latency.median()),
                "p95_latency_s": float(latency.quantile(0.95)),
                "p99_latency_s": float(latency.quantile(0.99)),
                "success_rate": float(grp["success"].mean()),
                "token_throughput_est": throughput,
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
