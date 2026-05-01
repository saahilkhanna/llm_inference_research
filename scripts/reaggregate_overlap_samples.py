#!/usr/bin/env python3
"""
Re-run grading + summaries on an incomplete run using only sample_ids that:
  completed successfully in every responses_*.jsonl under raw/, then either:
  - GSM8K slice: gsm8k_main_0 .. gsm8k_main_{cap-1} (default cap=300); or
  - LongBench / prefix slice:
      * intersection mode (default): successes ∩ IDs starting with the prefix,
        sorted, first --overlap-sample-cap; or
      * public_ordered: IDs from raw/public_samples.jsonl in file order matching
        the prefix, capped to --overlap-sample-cap (typically 150).

  With LongBench --impute-missing-llama-gold-mcq, any sample_id absent from
  responses_llama_cpp_*.jsonl is filled using the gold MCQ letter (cloned from the
  vLLM baseline row for metadata). Annotated as synthetic_llama_gold_imputation in
  the row — use only when you deliberately want analysis on a full rectangular grid.

Writes overlap metadata to processed/overlap_filter.json and overwrites processed/*.csv
and report/final_summary.{md,txt} for that run directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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


def _overlap_ids_longbench(
    raw_dir: Path,
    task_id_prefix: str,
    overlap_cap: int,
) -> tuple[set[str], dict]:
    """Intersection of successes, filter by sample_id prefix, then stable sort and take overlap_cap."""
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

    prefixed = sorted(sid for sid in inter if str(sid).startswith(task_id_prefix))
    capped = prefixed[:overlap_cap]
    keep = set(capped)
    return keep, {
        "per_file_success_counts": per_file,
        "intersection_all_success": len(inter),
        "after_prefix_only": len(prefixed),
        "overlap_sample_cap": overlap_cap,
        "after_overlap_cap": len(keep),
        "task_id_prefix": task_id_prefix,
        "longbench_sample_source": "intersection",
    }


def _public_ordered_longbench_ids(
    raw_dir: Path,
    task_id_prefix: str,
    overlap_cap: int,
) -> tuple[list[str], dict]:
    pub = raw_dir / "public_samples.jsonl"
    if not pub.is_file():
        raise FileNotFoundError(pub)
    ordered: list[str] = []
    with pub.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", "")).strip()
            if sid.startswith(task_id_prefix):
                ordered.append(sid)
    capped = ordered[: int(overlap_cap)]
    stats = {
        "per_file_success_counts": {},
        "from_public_samples_count": len(ordered),
        "overlap_sample_cap": overlap_cap,
        "after_cap": len(capped),
        "task_id_prefix": task_id_prefix,
        "longbench_sample_source": "public_ordered",
    }
    return capped, stats


def _response_jsonl_index(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", "")).strip()
            if sid:
                out[sid] = row
    return out


def _pick_llama_template_row(raw_dir: Path) -> dict | None:
    for p in sorted(raw_dir.glob("responses_llama_cpp*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
    return None


def _synthetic_llama_mcq_correct_row(
    tmpl_vllm: dict,
    llama_template: dict,
) -> dict:
    gold = str(tmpl_vllm.get("expected_answer", "")).strip()
    letter = ""
    if gold:
        letter = gold[0].upper()
        if letter not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"expected_answer MCQ letter missing/invalid sample_id="
                f"{tmpl_vllm.get('sample_id')}: {gold!r}"
            )
    r = copy.deepcopy(tmpl_vllm)
    r["backend"] = "llama_cpp"
    r["engine"] = llama_template.get("engine") or "llama_cpp"
    r["optimization_mode"] = "baseline"
    r["repeat_index"] = tmpl_vllm.get("repeat_index", 1)
    r["success"] = True
    r["error_message"] = ""
    rep = tmpl_vllm.get("repeat_index", 1)
    r["condition_fingerprint"] = f"llama_cpp|baseline|{rep}"
    r["endpoint_fingerprint"] = llama_template.get("endpoint_fingerprint") or tmpl_vllm.get(
        "endpoint_fingerprint", ""
    )
    r["response_text"] = f"Answer: {letter}" if letter else "Answer:"
    r["response_numeric_extracted"] = None
    r["synthetic_llama_gold_imputation"] = True
    r["latency_seconds"] = math.nan
    if "tti_seconds" in r:
        r["tti_seconds"] = math.nan
    if "ttft_seconds" in r:
        r["ttft_seconds"] = math.nan
    if "tpot_seconds" in r:
        r["tpot_seconds"] = math.nan
    if "tokens_per_second" in r:
        r["tokens_per_second"] = math.nan
    r["completion_tokens"] = None
    r["prompt_tokens"] = None
    r["total_tokens"] = None
    r["completion_tokens_est"] = 0
    r["estimated_cost_usd"] = None
    return r


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
        default=300,
        help="GSM8K mode: gsm8k_main_0 .. gsm8k_main_{cap-1} within overlap (ignored if --longbench-task-id-prefix set).",
    )
    parser.add_argument(
        "--longbench-task-id-prefix",
        type=str,
        default="",
        help="Non-empty enables LongBench overlap: successes ∩ ids starting with this prefix, sorted, first N via --overlap-sample-cap.",
    )
    parser.add_argument(
        "--overlap-sample-cap",
        type=int,
        default=120,
        help="LongBench mode: max samples after prefix filter (default 120).",
    )
    parser.add_argument(
        "--longbench-sample-source",
        choices=("intersection", "public_ordered"),
        default="intersection",
        help=(
            "LongBench only (when --longbench-task-id-prefix is set): intersection = successes ∩ "
            "prefix (default); public_ordered = first N IDs from raw/public_samples.jsonl for that prefix."
        ),
    )
    parser.add_argument(
        "--impute-missing-llama-gold-mcq",
        action="store_true",
        help=(
            "LongBench only: for responses_llama_cpp_*.jsonl, if a kept sample_id is missing, synthesize "
            "a successful row answering with the dataset gold MCQ letter (see module doc)."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    prefix = (args.longbench_task_id_prefix or "").strip()
    impute_llama_gold_mcq = bool(args.impute_missing_llama_gold_mcq)

    meta_source = "gsm8k_range"
    if prefix:
        if args.longbench_sample_source == "public_ordered":
            kept_ordered, stats = _public_ordered_longbench_ids(
                raw_dir, prefix, int(args.overlap_sample_cap)
            )
            files_all = sorted(raw_dir.glob("responses_*.jsonl"))
            pf: dict[str, int] = {}
            for path in files_all:
                pf[path.name] = len(_response_jsonl_index(path))
            stats["per_file_success_counts"] = pf
            meta_source = "longbench_public_ordered"
            keep_ids = set(kept_ordered)
        else:
            keep_ids, stats = _overlap_ids_longbench(raw_dir, prefix, int(args.overlap_sample_cap))
            kept_ordered = sorted(keep_ids)
            meta_source = "longbench_prefix"
    else:
        keep_ids, stats = _overlap_ids(raw_dir, args.gsm8k_cap)
        kept_ordered = sorted(
            keep_ids,
            key=lambda x: int(x.split("_")[-1]) if str(x.split("_")[-1]).isdigit() else x,
        )

    if not kept_ordered:
        raise RuntimeError(
            "No sample_ids after filtering; check public_samples or raw responses.")

    llama_template = _pick_llama_template_row(raw_dir) if (prefix and impute_llama_gold_mcq) else None
    if impute_llama_gold_mcq and llama_template is None:
        raise RuntimeError("--impute-missing-llama-gold-mcq requires at least one llama_cpp response row.")

    vllm_baseline_path = raw_dir / "responses_vllm_baseline_repeat_1.jsonl"
    vllm_template_by_sid: dict[str, dict] = {}
    if impute_llama_gold_mcq and prefix:
        vllm_template_by_sid = _response_jsonl_index(vllm_baseline_path)

    frames: list[pd.DataFrame] = []
    imputed_ids: list[str] = []

    for path in sorted(raw_dir.glob("responses_*.jsonl")):
        fname = path.name
        idx = _response_jsonl_index(path)
        is_llama = fname.startswith("responses_llama_cpp_")
        llama_impute_here = (
            bool(is_llama and impute_llama_gold_mcq and llama_template is not None)
        )
        rows: list[dict] = []

        if meta_source == "longbench_public_ordered" and not is_llama:
            first_bad: tuple[str, str | None] | None = None
            for sid in kept_ordered:
                rec = idx.get(sid)
                if rec is None or not rec.get("success"):
                    detail = (
                        None
                        if rec is None
                        else str(rec.get("error_message") or "")[:240] or "success=false"
                    )
                    first_bad = (sid, detail)
                    break
            if first_bad is not None:
                sid, detail = first_bad
                extras = "" if detail is None else f" ({detail})"
                raise RuntimeError(
                    f"{fname} missing or unsuccessful for public_ordered sample_id={sid}{extras}; "
                    f"cannot build a rectangular 150-sample grid."
                )

        if llama_impute_here:
            for sid in kept_ordered:
                if sid not in vllm_template_by_sid:
                    raise RuntimeError(
                        f"Need vLLM baseline row for {sid} when using "
                        "--impute-missing-llama-gold-mcq; missing in "
                        f"{vllm_baseline_path.name}"
                    )

        for sid in kept_ordered:
            row = idx.get(sid)
            if row is not None:
                rows.append(row)
                continue
            if llama_impute_here:
                rows.append(
                    _synthetic_llama_mcq_correct_row(
                        vllm_template_by_sid[sid], llama_template or {}
                    )
                )
                imputed_ids.append(sid)
                continue
            raise RuntimeError(
                f"{fname} has no sample_id={sid}. For rectangular LongBench grids, use "
                "--longbench-sample-source public_ordered with "
                "--impute-missing-llama-gold-mcq when llama_cpp stopped early."
            )

        if rows:
            frames.append(pd.DataFrame(rows))

    imputed_sorted = sorted(set(imputed_ids))
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("Merged filtered table is empty.")

    ensure_dir(run_dir / "processed")
    if meta_source == "longbench_public_ordered":
        kept_sorted = list(kept_ordered)
    elif prefix:
        kept_sorted = sorted(keep_ids)
    else:
        kept_sorted = sorted(
            keep_ids,
            key=lambda x: int(x.split("_")[-1]) if str(x.split("_")[-1]).isdigit() else x,
        )

    synth_warning: str | None = None
    if imputed_sorted:
        synth_warning = (
            "Synthetic llama_cpp rows use the dataset gold MCQ letter (not measured inference)."
        )

    meta = {
        "run_dir": str(run_dir),
        "filter_mode": meta_source,
        "gsm8k_cap": args.gsm8k_cap if not prefix else None,
        "kept_sample_count": len(keep_ids),
        "kept_sample_ids_sorted": kept_sorted,
        "synthetic_llama_gold_imputation_count": len(imputed_sorted),
        "synthetic_llama_gold_imputation_sample_ids": imputed_sorted,
        "warning_synthetic_llama_gold_imputation": synth_warning,
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

    summary = {"run_dir": str(run_dir), "kept_samples": len(keep_ids), **stats}
    summary["synthetic_llama_gold_imputation_count"] = len(imputed_sorted)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
