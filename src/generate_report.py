from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig
from .utils import ensure_dir


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def generate_final_report(config: AppConfig, run_dir: Path, run_id: str) -> Path:
    template_path = Path("report/final_summary_template.md")
    template = template_path.read_text(encoding="utf-8")

    failure_df = _safe_read_csv(run_dir / "processed" / "failure_bucket_summary.csv")
    latency_df = _safe_read_csv(run_dir / "processed" / "latency_summary.csv")
    public_cmp = _safe_read_csv(run_dir / "processed" / "public_per_sample_comparison.csv")
    custom_cmp = _safe_read_csv(run_dir / "processed" / "custom_per_sample_comparison.csv")
    cases_md = (run_dir / "case_studies" / "case_studies.md")
    case_text = cases_md.read_text(encoding="utf-8") if cases_md.exists() else "No case studies available."

    correctness_lines = []
    for label, df in [("public", public_cmp), ("custom", custom_cmp)]:
        if df.empty:
            correctness_lines.append(f"- {label}: no comparisons available")
            continue
        buckets = df["bucket"].value_counts().to_dict()
        correctness_lines.append(f"- {label}: {buckets}")

    failure_lines = ["- " + f"{row['bucket']}: {int(row['count'])}" for _, row in failure_df.iterrows()] if not failure_df.empty else ["- No failure bucket summary available"]
    latency_lines = []
    if latency_df.empty:
        latency_lines.append("- No latency summary available")
    else:
        for _, row in latency_df.iterrows():
            latency_lines.append(
                f"- {row['backend']} | {row['workload']} | {row['workload_class']}: "
                f"mean={row['mean_latency_s']:.3f}s, p95={row['p95_latency_s']:.3f}s, success={row['success_rate']:.2%}"
            )

    artifacts = [
        f"- `{run_dir / 'manifest.json'}`",
        f"- `{run_dir / 'tool_selection_memo.md'}`",
        f"- `{run_dir / 'raw'}`",
        f"- `{run_dir / 'processed/public_per_sample_comparison.csv'}`",
        f"- `{run_dir / 'processed/custom_per_sample_comparison.csv'}`",
        f"- `{run_dir / 'processed/failure_bucket_summary.csv'}`",
        f"- `{run_dir / 'processed/latency_summary.csv'}`",
        f"- `{run_dir / 'case_studies/case_studies.csv'}`",
        f"- `{run_dir / 'case_studies/case_studies.md'}`",
        f"- `{run_dir / 'report/final_summary.md'}`",
        f"- `{run_dir / 'logs'}`",
    ]

    filled = (
        template.replace(
            "{{TOOL_STACK}}",
            "- LightEval for benchmark-oriented correctness and sample details\n"
            "- Custom grading and failure bucketing for black-box comparison\n"
            "- AIPerf for request-level performance exports",
        )
        .replace(
            "{{WORKLOADS}}",
            "- Public: HumanEval-style coding prompts, GSM8K, competition math, and a long-context public-style slice\n"
            "- Custom: mixed short/medium/long Edge-IoT data-analysis prompts\n"
            "- Related motivation: silent correctness issues in optimized inference systems (MDPI basis provided by user).",
        )
        .replace("{{SETUP}}", f"- Model ID: `{config.model_id}`\n- Run ID: `{run_id}`\n- Endpoint strategy: `{config.shutdown_mode}`")
        .replace("{{CORRECTNESS_SUMMARY}}", "\n".join(correctness_lines))
        .replace("{{FAILURE_BUCKET_SUMMARY}}", "\n".join(failure_lines))
        .replace("{{PERFORMANCE_SUMMARY}}", "\n".join(latency_lines))
        .replace("{{CASE_STUDIES}}", case_text)
        .replace(
            "{{LIMITATIONS_EXTRA}}",
            "- Endpoint cold starts, rate limits, and request formatting differences can affect observed behavior.\n"
            "- Synthetic performance traces are informative for latency stress but do not prove benchmark correctness properties.",
        )
        .replace(
            "{{FUTURE_WORK}}",
            "- Add self-hosted backend instrumentation for controlled ablations.\n"
            "- Expand long-context public tasks when budget permits.\n"
            "- Add repeated-seed confidence intervals for failure buckets.",
        )
        .replace("{{ARTIFACTS}}", "\n".join(artifacts))
    )

    report_dir = run_dir / "report"
    ensure_dir(report_dir)
    out_path = report_dir / "final_summary.md"
    out_path.write_text(filled, encoding="utf-8")
    txt_path = report_dir / "final_summary.txt"
    txt_path.write_text(filled, encoding="utf-8")
    return out_path
