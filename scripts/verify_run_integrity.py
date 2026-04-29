#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _latest_run_dir(results_dir: Path, run_name: str) -> Path:
    matches = sorted(results_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for run_name='{run_name}' under {results_dir}")
    return matches[-1]


def verify(run_dir: Path) -> None:
    logs = run_dir / "logs"
    processed = run_dir / "processed"
    raw = run_dir / "raw"

    endpoint_resolution = logs / "endpoint_resolution.json"
    if endpoint_resolution.exists():
        endpoint_data = json.loads(endpoint_resolution.read_text(encoding="utf-8"))
        conditions = endpoint_data.get("conditions", [])
        if not conditions:
            raise RuntimeError("endpoint_resolution.json has no resolved conditions")
    else:
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            raise RuntimeError("Missing logs/endpoint_resolution.json and manifest.json")
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        endpoint_map = manifest_data.get("endpoints", {}) or {}
        conditions = list(endpoint_map.values())
        if not conditions:
            raise RuntimeError("No endpoint resolution data found in endpoint_resolution.json or manifest.json")

    raw_files = sorted(raw.glob("responses_*_repeat_*.jsonl"))
    if not raw_files:
        raw_files = sorted(raw.glob("responses_*.jsonl"))
    if not raw_files:
        raise RuntimeError("No condition-scoped raw response files found")

    graded_path = processed / "all_samples_graded.csv"
    if not graded_path.exists():
        raise RuntimeError("Missing processed/all_samples_graded.csv")
    graded = pd.read_csv(graded_path)
    if graded.empty:
        raise RuntimeError("processed/all_samples_graded.csv is empty")

    sample_ids_all = set(graded["sample_id"].astype(str).tolist())
    for (backend, mode, rep), grp in graded.groupby(["backend", "optimization_mode", "repeat_index"]):
        ids = set(grp["sample_id"].astype(str).tolist())
        if ids != sample_ids_all:
            raise RuntimeError(
                f"Sample ID mismatch in graded outputs for condition {backend}:{mode}:repeat_{int(rep)}"
            )

    print(f"[verify_run_integrity] OK: {run_dir}")
    print(f"[verify_run_integrity] conditions={len(conditions)} raw_files={len(raw_files)} rows={len(graded)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify integrity artifacts for a completed run.")
    parser.add_argument("--run-dir", default="", help="Absolute or relative run directory.")
    parser.add_argument("--results-dir", default="results", help="Results root for latest-run lookup.")
    parser.add_argument("--run-name", default="", help="Run name used to find latest run in --results-dir.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if not args.run_name:
            raise ValueError("Provide --run-dir, or provide both --results-dir and --run-name.")
        run_dir = _latest_run_dir(Path(args.results_dir), args.run_name)

    verify(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
