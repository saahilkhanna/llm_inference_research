from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def _jsonl_to_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def build_parsed_table(run_dir: Path) -> pd.DataFrame:
    response_files = sorted((run_dir / "raw").glob("responses_*.jsonl"))
    frames = [_jsonl_to_df(path) for path in response_files]
    frames = [frame for frame in frames if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        return merged

    ensure_dir(run_dir / "processed")
    merged.to_csv(run_dir / "processed" / "all_samples_raw.csv", index=False)
    return merged
