#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_if_nonempty(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    df.to_csv(path, index=False)


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def organize_backend(run_dir: Path, backend: str, output_dir: Path) -> None:
    manifest = _load_manifest(run_dir)
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    endpoint_map = manifest.get("endpoints", {}) if isinstance(manifest, dict) else {}
    processed = run_dir / "processed"
    graded = _safe_read_csv(processed / "all_samples_graded.csv")
    if graded.empty:
        raise RuntimeError(f"Missing or empty graded data at {processed / 'all_samples_graded.csv'}")

    backend_rows = graded[graded["backend"] == backend].copy()
    if backend_rows.empty:
        raise RuntimeError(f"No rows found for backend={backend} in {run_dir}")

    base_out = output_dir / backend
    for mode, group in backend_rows.groupby("optimization_mode", dropna=False):
        mode_name = str(mode) if str(mode) else "unknown_mode"
        mode_out = base_out / mode_name
        mode_out.mkdir(parents=True, exist_ok=True)
        group.to_csv(mode_out / "all_samples_graded.csv", index=False)

        for name in ("public_per_sample_comparison.csv", "latency_summary.csv", "failure_bucket_summary.csv"):
            df = _safe_read_csv(processed / name)
            if df.empty:
                continue
            if "backend" in df.columns:
                df = df[df["backend"] == backend]
            if "optimization_mode" in df.columns:
                df = df[df["optimization_mode"] == mode]
            _write_if_nonempty(df, mode_out / name)

        key = f"{backend}:{mode_name}"
        endpoint_info = endpoint_map.get(key, {})
        benchmark_meta = {
            "run_id": manifest.get("run_id"),
            "backend": backend,
            "optimization_mode": mode_name,
            "public_benchmark_enabled": config.get("public_benchmark_enabled"),
            "public_task_ids": config.get("public_task_ids", []),
            "public_tasks": config.get("public_tasks", []),
            "standard_eval_enabled": config.get("standard_eval_enabled"),
            "standard_evaluator": config.get("standard_evaluator"),
            "standard_eval_tasks": config.get("standard_eval_tasks", []),
            "standard_eval_limit_override": config.get("standard_eval_limit_override"),
            "limit": config.get("limit"),
        }
        model_meta = {
            "run_id": manifest.get("run_id"),
            "backend": backend,
            "optimization_mode": mode_name,
            "model_id": endpoint_info.get("model_id", config.get(f"{backend}_model_id", config.get("model_id"))),
            "endpoint_name": endpoint_info.get("name"),
            "endpoint_url": endpoint_info.get("url"),
            "endpoint_args": endpoint_info.get("endpoint_args", []),
            "framework": config.get(f"{backend}_endpoint_framework"),
            "engine_image_url": config.get(f"{backend}_engine_image_url"),
            "instance_type": config.get("endpoint_instance_type"),
        }
        _write_json(mode_out / "benchmark_config.json", benchmark_meta)
        _write_json(mode_out / "model_config.json", model_meta)
        _write_json(mode_out / "run_manifest_excerpt.json", {"run_id": manifest.get("run_id"), "status": manifest.get("status")})


def rebuild_aggregate(output_dir: Path) -> None:
    rows: list[pd.DataFrame] = []
    bench_rows: list[dict] = []
    model_rows: list[dict] = []
    for graded_path in output_dir.glob("*/*/all_samples_graded.csv"):
        df = _safe_read_csv(graded_path)
        if df.empty:
            continue
        rows.append(df)
    for p in output_dir.glob("*/*/benchmark_config.json"):
        try:
            bench_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    for p in output_dir.glob("*/*/model_config.json"):
        try:
            model_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    agg_out = output_dir / "aggregated_results"
    agg_out.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(agg_out / "aggregated_results.csv", index=False)
    latency_col = "latency_ms" if "latency_ms" in all_df.columns else "latency_seconds"
    if latency_col not in all_df.columns:
        all_df[latency_col] = 0.0
    if "error_flag" in all_df.columns:
        error_series = all_df["error_flag"].astype(float)
    elif "success" in all_df.columns:
        error_series = (~all_df["success"].fillna(False).astype(bool)).astype(float)
        all_df["error_flag"] = error_series
    else:
        error_series = pd.Series([0.0] * len(all_df))
        all_df["error_flag"] = error_series
    if "correctness" in all_df.columns:
        c = all_df["correctness"].astype(str).str.lower()
        all_df["correct_binary"] = c.map({"correct": 1.0, "wrong": 0.0})
        all_df["missing_correctness"] = all_df["correct_binary"].isna()
    else:
        all_df["correct_binary"] = 0.0
        all_df["missing_correctness"] = True
    summary = (
        all_df.groupby(["backend", "optimization_mode"], dropna=False)
        .agg(
            rows=("sample_id", "count"),
            correctness_mean=("correct_binary", "mean"),
            missing_correctness_rows=("missing_correctness", "sum"),
            error_rate=("error_flag", "mean"),
            p95_latency=("latency_seconds", lambda s: float(s.quantile(0.95)) if len(s) else 0.0)
            if latency_col == "latency_seconds"
            else ("latency_ms", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
        .reset_index()
    )
    if "estimated_cost_usd" in all_df.columns:
        cost = all_df.groupby(["backend", "optimization_mode"], dropna=False)["estimated_cost_usd"].sum().reset_index(name="total_estimated_cost_usd")
        summary = summary.merge(cost, on=["backend", "optimization_mode"], how="left")
    else:
        summary["total_estimated_cost_usd"] = None
    summary["cost_per_100_correct_est"] = summary.apply(
        lambda r: (float(r["total_estimated_cost_usd"]) / max(1e-9, float(r["correctness_mean"]) * float(r["rows"])) * 100.0)
        if pd.notna(r["total_estimated_cost_usd"]) and pd.notna(r["correctness_mean"]) and float(r["correctness_mean"]) > 0
        else None,
        axis=1,
    )
    summary.to_csv(agg_out / "aggregated_summary.csv", index=False)
    _write_json(agg_out / "benchmark_coverage.json", {"entries": bench_rows})
    _write_json(agg_out / "model_matrix.json", {"entries": model_rows})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend", required=True, choices=["llama_cpp", "vllm", "sglang"])
    parser.add_argument("--output-dir", default="final_summary")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    organize_backend(run_dir=run_dir, backend=args.backend, output_dir=output_dir)
    rebuild_aggregate(output_dir=output_dir)
    print(f"[organize_final_summary] updated backend={args.backend} from {run_dir}")
    print(f"[organize_final_summary] aggregate={output_dir / 'aggregated_results' / 'aggregated_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
