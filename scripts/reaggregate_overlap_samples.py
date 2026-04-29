#!/usr/bin/env python3
"""
Re-run grading + summaries on an incomplete run using only sample_ids that:
  - completed successfully in every responses_*.jsonl under raw/, and
  - fall in gsm8k_main_0 .. gsm8k_main_{cap-1} (default cap=150).

Writes overlap metadata to processed/overlap_filter.json and overwrites processed/*.csv
and report/final_summary.{md,txt} for that run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.compare_backends import compare_backends
from src.config import load_config
from src.generate_report import generate_final_report
from src.grade_outputs import grade_outputs
from src.integrity_checks import validate_post_run_integrity
from src.select_case_studies import select_case_studies
from src.summarize_metrics import summarize_latency
from src.utils import ensure_dir


def _jsonl_success_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("success"):
                sid = str(row.get("sample_id", "")).strip()
                if sid:
                    ids.add(sid)
    return ids


def _overlap_ids(raw_dir: Path, gsm8k_cap: int) -> tuple[set[str], dict[str, int]]:
    files = sorted(raw_dir.glob("responses_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No responses_*.jsonl under {raw_dir}")

    per_file: dict[str, int] = {}
    sets: list[set[str]] = []
    for p in files:
        s = _jsonl_success_ids(p)
        sets.append(s)
        per_file[p.name] = len(s)

    inter = sets[0].copy()
    for s in sets[1:]:
        inter &= s

    cap_ids = {f"gsm8k_main_{i}" for i in range(gsm8k_cap)}
    keep = inter & cap_ids
    return keep, {"per_file_success_counts": per_file, "intersection_before_cap": len(inter), "after_gsm8k_cap": len(keep)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/20260429T230230Z_black_box_vllm_sglang"),
        help="Run directory containing raw/responses_*.jsonl",
    )
    parser.add_argument(
        "--gsm8k-cap",
        type=int,
        default=150,
        help="Keep only gsm8k_main_0 .. gsm8k_main_{cap-1} within the overlap (default 150).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    keep_ids, stats = _overlap_ids(raw_dir, args.gsm8k_cap)
    if not keep_ids:
        raise RuntimeError("No overlapping successful sample_ids after filtering; check raw responses.")

    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("responses_*.jsonl")):
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("sample_id", "")).strip() in keep_ids:
                    rows.append(row)
        if rows:
            frames.append(pd.DataFrame(rows))

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("Merged filtered table is empty.")

    ensure_dir(run_dir / "processed")
    meta = {
        "run_dir": str(run_dir),
        "gsm8k_cap": args.gsm8k_cap,
        "kept_sample_count": len(keep_ids),
        "kept_sample_ids_sorted": sorted(keep_ids, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x),
        **stats,
    }
    (run_dir / "processed" / "overlap_filter.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    merged.to_csv(run_dir / "processed" / "all_samples_raw.csv", index=False)
    graded = grade_outputs(merged)

    conditions = sorted(
        {(str(r["optimization_mode"]), int(r["repeat_index"]), str(r["backend"])) for _, r in graded.iterrows()},
        key=lambda t: (t[2], t[0], t[1]),
    )

    config = load_config()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mc = manifest.get("config") or {}
        if isinstance(mc, dict) and "standard_eval_enabled" in mc:
            config.standard_eval_enabled = bool(mc["standard_eval_enabled"])

    validate_post_run_integrity(config=config, run_dir=run_dir, graded=graded, conditions=conditions)

    graded.to_csv(run_dir / "processed" / "all_samples_graded.csv", index=False)
    compare_backends(graded, run_dir)
    select_case_studies(run_dir, per_bucket=int(config.analysis.get("case_studies_per_bucket", 3)))
    summarize_latency(graded, run_dir)
    run_id = run_dir.name
    generate_final_report(config, run_dir, run_id)

    print(json.dumps({"run_dir": str(run_dir), "kept_samples": len(keep_ids), **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
