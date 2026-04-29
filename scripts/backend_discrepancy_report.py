#!/usr/bin/env python3
"""
Aggregate backend disagreement stats and list discrepancy sample_ids for a completed run.

Reads processed/public_per_sample_comparison.csv and processed/all_samples_graded.csv
from --run-dir and writes:
  - processed/discrepancy_aggregate.csv
  - processed/discrepancy_samples_baseline.csv (llama vs vllm vs sglang, baseline only)
  - processed/discrepancy_samples_by_mode.csv (per-sample correctness wide table)
  - report/backend_discrepancy_summary.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


def _norm_correctness(x: object) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"correct", "wrong", "unknown"} else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/20260429T230230Z_black_box_vllm_sglang"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    proc = run_dir / "processed"
    cmp_path = proc / "public_per_sample_comparison.csv"
    graded_path = proc / "all_samples_graded.csv"
    if not cmp_path.exists():
        raise FileNotFoundError(cmp_path)
    if not graded_path.exists():
        raise FileNotFoundError(graded_path)

    cmp_df = pd.read_csv(cmp_path)
    graded = pd.read_csv(graded_path)

    # --- vLLM vs SGLang (existing bucket column), per optimization_mode ---
    agg_rows = []
    if not cmp_df.empty and "bucket" in cmp_df.columns:
        for mode in sorted(cmp_df["optimization_mode"].dropna().unique()):
            sub = cmp_df[cmp_df["optimization_mode"] == mode]
            vc = sub["bucket"].value_counts()
            for bucket, count in vc.items():
                agg_rows.append(
                    {
                        "slice": "vllm_vs_sglang",
                        "optimization_mode": mode,
                        "metric": str(bucket),
                        "count": int(count),
                    }
                )
            agg_rows.append(
                {
                    "slice": "vllm_vs_sglang",
                    "optimization_mode": mode,
                    "metric": "_total_rows",
                    "count": int(len(sub)),
                }
            )

    # --- Wide correctness: sample_id × optimization_mode ---
    wide_rows = []
    for (sid, om, rep), grp in graded.groupby(["sample_id", "optimization_mode", "repeat_index"], dropna=False):
        d = {str(r["backend"]): _norm_correctness(r["correctness"]) for _, r in grp.iterrows()}
        wide_rows.append(
            {
                "sample_id": sid,
                "optimization_mode": om,
                "repeat_index": int(rep),
                "llama_cpp": d.get("llama_cpp", ""),
                "vllm": d.get("vllm", ""),
                "sglang": d.get("sglang", ""),
            }
        )
    wide = pd.DataFrame(wide_rows)

    # Baseline triple patterns (llama participates only here)
    base = wide[wide["optimization_mode"] == "baseline"].copy()
    patterns = []
    for _, r in base.iterrows():
        cl, cv, cs = r["llama_cpp"], r["vllm"], r["sglang"]
        if not cv or not cs:
            continue
        # llama may be missing if engine omitted from run
        triple_label = ""
        if cl:
            parts = sorted([(cl, "L"), (cv, "V"), (cs, "S")])
            triple_label = f"L={cl}_V={cv}_S={cs}"
        patterns.append(
            {
                "sample_id": r["sample_id"],
                "optimization_mode": "baseline",
                "correctness_llama_cpp": cl or "(none)",
                "correctness_vllm": cv,
                "correctness_sglang": cs,
                "triple_pattern": triple_label,
                "vllm_disagrees_sglang": cv != cs,
                "llama_disagrees_vllm": bool(cl) and cl != cv,
                "llama_disagrees_sglang": bool(cl) and cl != cs,
            }
        )
    base_patterns = pd.DataFrame(patterns)

    triple_counts = base_patterns["triple_pattern"].value_counts().reset_index()
    triple_counts.columns = ["triple_pattern", "count"]
    triple_counts = triple_counts[triple_counts["triple_pattern"].astype(str).str.len() > 0]

    if not base_patterns.empty:
        for label, mask in [
            ("baseline_vllm_not_sglang", base_patterns["vllm_disagrees_sglang"]),
            ("baseline_llama_not_vllm", base_patterns["llama_disagrees_vllm"]),
            ("baseline_llama_not_sglang", base_patterns["llama_disagrees_sglang"]),
            ("baseline_any_pair_mismatch", base_patterns["vllm_disagrees_sglang"] | base_patterns["llama_disagrees_vllm"] | base_patterns["llama_disagrees_sglang"]),
        ]:
            agg_rows.append(
                {
                    "slice": label,
                    "optimization_mode": "baseline",
                    "metric": "count",
                    "count": int(mask.sum()),
                }
            )

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(proc / "discrepancy_aggregate.csv", index=False)
    wide.to_csv(proc / "discrepancy_samples_by_mode.csv", index=False)
    base_patterns.to_csv(proc / "discrepancy_samples_baseline.csv", index=False)
    triple_counts.to_csv(proc / "discrepancy_triple_pattern_counts.csv", index=False)

    # Markdown report
    lines = [
        "# Backend discrepancy summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## vLLM vs SGLang (`public_per_sample_comparison` buckets)",
        "",
    ]
    if not cmp_df.empty:
        for mode in sorted(cmp_df["optimization_mode"].dropna().unique()):
            sub = cmp_df[cmp_df["optimization_mode"] == mode]
            lines.append(f"### optimization_mode = `{mode}`")
            lines.append("")
            lines.append("| bucket | count |")
            lines.append("|--------|-------|")
            for bucket, count in sub["bucket"].value_counts().items():
                lines.append(f"| {bucket} | {count} |")
            lines.append("")
            disc = sub[sub["bucket"].isin(["vllm_correct_sglang_wrong", "sglang_correct_vllm_wrong"])]
            ids_v_only = disc[disc["bucket"] == "vllm_correct_sglang_wrong"]["sample_id"].tolist()
            ids_s_only = disc[disc["bucket"] == "sglang_correct_vllm_wrong"]["sample_id"].tolist()
            lines.append(f"- **vLLM correct, SGLang wrong** ({len(ids_v_only)}): `{ids_v_only}`")
            lines.append(f"- **SGLang correct, vLLM wrong** ({len(ids_s_only)}): `{ids_s_only}`")
            lines.append("")

    lines.extend(
        [
            "## Baseline: llama.cpp vs vLLM vs SGLang",
            "",
            "Triple correctness counts:",
            "",
            "| pattern (L=llama, V=vLLM, S=SGLang) | count |",
            "|---------------------------------------|-------|",
        ]
    )
    for _, tr in triple_counts.iterrows():
        lines.append(f"| `{tr['triple_pattern']}` | {int(tr['count'])} |")
    lines.append("")

    if not base_patterns.empty:
        any_mismatch = (
            base_patterns["vllm_disagrees_sglang"]
            | base_patterns["llama_disagrees_vllm"]
            | base_patterns["llama_disagrees_sglang"]
        )
        n_any = int(any_mismatch.sum())
        lines.append(f"- Rows with **any** pairwise disagreement among engines that have data: **{n_any}** / {len(base_patterns)}")
        lines.append("")
        lines.append("### Samples where vLLM and SGLang disagree (baseline)")
        lines.append("")
        v_s = base_patterns[base_patterns["vllm_disagrees_sglang"]]["sample_id"].tolist()
        lines.append(", ".join(f"`{x}`" for x in v_s) if v_s else "(none)")
        lines.append("")
        lines.append("### Samples where llama.cpp differs from vLLM (baseline)")
        lines.append("")
        l_v = base_patterns[base_patterns["llama_disagrees_vllm"]]["sample_id"].tolist()
        lines.append(", ".join(f"`{x}`" for x in l_v) if l_v else "(none)")
        lines.append("")
        lines.append("### Samples where llama.cpp differs from SGLang (baseline)")
        lines.append("")
        l_s = base_patterns[base_patterns["llama_disagrees_sglang"]]["sample_id"].tolist()
        lines.append(", ".join(f"`{x}`" for x in l_s) if l_s else "(none)")
        lines.append("")

    report_path = run_dir / "report" / "backend_discrepancy_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {proc / 'discrepancy_aggregate.csv'}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
