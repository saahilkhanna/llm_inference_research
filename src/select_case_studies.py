from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .normalize_outputs import normalize_text
from .utils import ensure_dir


def _normalized_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].fillna("").astype(str).map(normalize_text)


def _select_rows(df: pd.DataFrame, mask, n: int, label: str) -> pd.DataFrame:
    subset = df[mask].head(n).copy()
    if subset.empty:
        return subset
    subset["case_type"] = label
    return subset


def select_case_studies(run_dir: Path, per_bucket: int = 3) -> dict[str, Path]:
    public_path = run_dir / "processed" / "public_per_sample_comparison.csv"
    custom_path = run_dir / "processed" / "custom_per_sample_comparison.csv"
    df_list = []
    for path in (public_path, custom_path):
        if path.exists():
            try:
                df = pd.read_csv(path)
            except EmptyDataError:
                continue
            if not df.empty:
                df_list.append(df)
    merged = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    ensure_dir(run_dir / "case_studies")
    csv_path = run_dir / "case_studies" / "case_studies.csv"
    md_path = run_dir / "case_studies" / "case_studies.md"

    if merged.empty:
        pd.DataFrame(columns=["case_type"]).to_csv(csv_path, index=False)
        md_path.write_text("# Case Studies\n\nNo case studies available.\n", encoding="utf-8")
        return {"csv": csv_path, "md": md_path}

    both_correct_diff = merged[
        (merged.get("bucket") == "both_correct")
        & (_normalized_column(merged, "response_text_vllm") != _normalized_column(merged, "response_text_sglang"))
    ]

    studies = pd.concat(
        [
            _select_rows(merged, merged.get("bucket") == "both_wrong", per_bucket, "both_wrong"),
            _select_rows(
                merged, merged.get("bucket") == "vllm_correct_sglang_wrong", per_bucket, "vllm_only_correct"
            ),
            _select_rows(
                merged, merged.get("bucket") == "sglang_correct_vllm_wrong", per_bucket, "sglang_only_correct"
            ),
            _select_rows(
                merged, (merged.get("workload_class") == "long") & (merged.get("bucket") != "both_correct"), per_bucket,
                "long_context_failure",
            ),
            _select_rows(both_correct_diff, both_correct_diff.index == both_correct_diff.index, per_bucket, "both_correct_but_different"),
        ],
        ignore_index=True,
    )
    studies = studies.drop_duplicates(subset=["sample_id", "case_type"], keep="first")
    studies.to_csv(csv_path, index=False)

    lines = [
        "# Representative Case Studies",
        "",
        "These cases illustrate observable response-level patterns. Causal claims about internal GPU/KV/scheduler behavior are not supported by this data alone.",
        "",
    ]
    for _, row in studies.iterrows():
        lines.extend(
            [
                f"## {row.get('case_type', 'case')} - {row.get('sample_id', 'unknown')}",
                f"- Task/workload: `{row.get('task_id', 'unknown')}` / `{row.get('workload_class', 'unknown')}`",
                f"- Bucket: `{row.get('bucket', 'unknown')}`",
                "- Interpretation: One possible explanation is output-format or retrieval behavior differences under identical prompts; we cannot prove internal backend mechanisms from endpoint outputs alone.",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "md": md_path}
