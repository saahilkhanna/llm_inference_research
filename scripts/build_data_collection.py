#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _jsonl_to_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return pd.DataFrame(rows)


def _flatten_manifest(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not manifest_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(data.get("run_id", manifest_path.parent.name))
    manifest_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "mode": data.get("mode"),
                "status": data.get("status"),
                "timestamp_utc": data.get("timestamp_utc"),
                "error": data.get("error"),
            }
        ]
    )
    config = data.get("config", {}) if isinstance(data.get("config", {}), dict) else {}
    config_df = pd.DataFrame([{**{"run_id": run_id}, **config}]) if config else pd.DataFrame()
    endpoints = data.get("endpoints", {}) if isinstance(data.get("endpoints", {}), dict) else {}
    endpoint_rows = []
    for key, info in endpoints.items():
        if isinstance(info, dict):
            endpoint_rows.append({"run_id": run_id, "endpoint_key": key, **info})
    endpoints_df = pd.DataFrame(endpoint_rows)
    return manifest_df, config_df, endpoints_df


def _write_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return 0
    df.to_csv(path, index=False)
    return int(len(df))


def _collect_run(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    run_id = run_dir.name
    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict] = []

    def emit(name: str, df: pd.DataFrame) -> None:
        rows = _write_csv(df, out / f"{name}.csv")
        inventory_rows.append({"run_id": run_id, "table": name, "rows": rows, "missing": rows == 0})

    processed = run_dir / "processed"
    logs = run_dir / "logs"

    # Core processed tables
    graded = _safe_read_csv(processed / "all_samples_graded.csv")
    if not graded.empty:
        graded["run_id"] = run_id
        if "benchmark_valid_for_claims" in graded.columns:
            graded["missing_for_claims"] = ~graded["benchmark_valid_for_claims"].fillna(False).astype(bool)
        if "correctness" in graded.columns:
            c = graded["correctness"].astype(str).str.lower()
            graded["missing_correctness"] = ~c.isin(["correct", "wrong"])
    emit("all_samples_graded", graded)

    raw_df = _safe_read_csv(processed / "all_samples_raw.csv")
    if raw_df.empty:
        frames = [_jsonl_to_df(p) for p in sorted((run_dir / "raw").glob("responses_*.jsonl"))]
        frames = [f for f in frames if not f.empty]
        raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not raw_df.empty:
        raw_df["run_id"] = run_id
    emit("all_samples_raw", raw_df)

    for name in [
        "public_per_sample_comparison",
        "failure_bucket_summary",
        "latency_summary",
        "all_samples_standard_eval",
        "all_samples_comparison",
    ]:
        df = _safe_read_csv(processed / f"{name}.csv")
        if not df.empty:
            df["run_id"] = run_id
        emit(name, df)

    # Manifest/config/endpoint metadata
    manifest_df, config_df, endpoints_df = _flatten_manifest(run_dir / "manifest.json")
    emit("run_manifest", manifest_df)
    emit("run_config_flat", config_df)
    emit("endpoint_manifest", endpoints_df)
    resolution = _safe_read_csv(logs / "endpoint_resolution.csv")
    if resolution.empty:
        resolution_json = logs / "endpoint_resolution.json"
        if resolution_json.exists():
            payload = json.loads(resolution_json.read_text(encoding="utf-8"))
            conds = payload.get("conditions", [])
            resolution = pd.DataFrame([{"run_id": run_id, **c} for c in conds if isinstance(c, dict)])
    else:
        resolution["run_id"] = run_id
    emit("endpoint_resolution", resolution)

    # Log/event tables
    for name in [
        "standard_evaluator",
        "run_progress",
        "condition_progress",
        "integrity_checks",
        "endpoint_manager",
        "endpoint_shutdown",
        "aiperf",
    ]:
        df = _jsonl_to_df(logs / f"{name}.jsonl")
        if not df.empty:
            df["run_id"] = run_id
        emit(f"{name}_events", df)

    inv = pd.DataFrame(inventory_rows)
    emit("file_inventory", inv)
    return inv


def _load_runs(results_dir: Path, run_dir: Path | None = None) -> Iterable[Path]:
    if run_dir is not None:
        return [run_dir]
    out = []
    for manifest in sorted(results_dir.glob("*/manifest.json")):
        out.append(manifest.parent)
    return out


def _aggregate(output_dir: Path) -> None:
    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    by_table: dict[str, list[pd.DataFrame]] = {}
    for run_folder in output_dir.iterdir():
        if not run_folder.is_dir() or run_folder.name == "aggregated":
            continue
        for csv_path in run_folder.glob("*.csv"):
            name = csv_path.stem
            df = _safe_read_csv(csv_path)
            if df.empty:
                continue
            by_table.setdefault(name, []).append(df)
    for name, parts in by_table.items():
        merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if merged.empty:
            continue
        merged.to_csv(agg_dir / f"{name}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comprehensive CSV data collection bundle.")
    parser.add_argument("--results-dir", default="real_results/USED_RESULTS")
    parser.add_argument("--output-dir", default="final_summary/data_collection")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)
    run_paths = list(_load_runs(results_dir=results_dir, run_dir=run_dir))
    if not run_paths:
        raise RuntimeError(f"No run directories found under {results_dir}")
    for rdir in run_paths:
        _collect_run(rdir, output_dir)
    _aggregate(output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
