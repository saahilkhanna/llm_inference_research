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
                "mean_ttft_s",
                "median_ttft_s",
                "p95_ttft_s",
                "ttft_missing_rate",
                "mean_tpot_s",
                "median_tpot_s",
                "mean_prompt_tokens",
                "mean_completion_tokens",
                "token_counts_missing_rate",
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
        ttft = pd.to_numeric(grp.get("ttft_seconds", pd.Series(dtype=float)), errors="coerce")
        tpot = pd.to_numeric(grp.get("tpot_seconds", pd.Series(dtype=float)), errors="coerce")
        prompt_tokens = pd.to_numeric(grp.get("prompt_tokens", grp.get("prompt_tokens_est", pd.Series(dtype=float))), errors="coerce")
        completion_tokens = pd.to_numeric(
            grp.get("completion_tokens", grp.get("completion_tokens_est", pd.Series(dtype=float))), errors="coerce"
        )
        token_missing = grp.get("token_counts_missing", False)
        if isinstance(token_missing, pd.Series):
            token_missing_rate = float(token_missing.fillna(True).astype(bool).mean())
        else:
            token_missing_rate = float(token_missing)
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
                "mean_ttft_s": float(ttft.mean()) if not ttft.dropna().empty else None,
                "median_ttft_s": float(ttft.median()) if not ttft.dropna().empty else None,
                "p95_ttft_s": float(ttft.quantile(0.95)) if not ttft.dropna().empty else None,
                "ttft_missing_rate": float(ttft.isna().mean()) if len(ttft) else 1.0,
                "mean_tpot_s": float(tpot.mean()) if not tpot.dropna().empty else None,
                "median_tpot_s": float(tpot.median()) if not tpot.dropna().empty else None,
                "mean_prompt_tokens": float(prompt_tokens.mean()) if not prompt_tokens.dropna().empty else None,
                "mean_completion_tokens": float(completion_tokens.mean()) if not completion_tokens.dropna().empty else None,
                "token_counts_missing_rate": token_missing_rate,
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
