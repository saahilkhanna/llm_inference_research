from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .utils import ensure_dir


def generate_tool_selection_memo(config: AppConfig, run_dir: Path) -> Path:
    memo_path = run_dir / "tool_selection_memo.md"
    ensure_dir(memo_path.parent)

    content = f"""# Tool Selection Memo

## 1) Best tool for public benchmark correctness
**Choice:** Hugging Face LightEval  
**Why:** It supports benchmark-style evaluation with objective metrics, multiple backend integrations, and `save_details` sample-level output artifacts.

## 2) Best tool for sample-level outputs and failure analysis
**Choice:** LightEval + custom pandas post-processing  
**Why:** LightEval provides detailed per-sample records, while custom analysis enforces project-specific failure buckets (`both_correct`, `both_wrong`, `vllm_correct_sglang_wrong`, etc.) and representative case selection.

## 3) Best tool for latency/performance measurement
**Choice:** AIPerf  
**Why:** It exports per-request and aggregate metrics, including request latency distributions and throughput-oriented fields in CSV/JSON/JSONL formats.

## 4) Easiest tool with Hugging Face Endpoints
**Choice:** LightEval and direct HTTP runners  
**Why:** LightEval has dedicated endpoint support, and direct request runners are simple for identical black-box prompt replay across two endpoints.

## 5) Easiest tool with vLLM and SGLang
**Choice:** HTTP-compatible request runner + LightEval-inspired task formatting  
**Why:** Using endpoint URLs with the same request schema avoids backend-specific internals and keeps comparison black-box and reproducible.

## 6) Chosen stack and why
- **Public correctness:** LightEval-style task selection + cached public samples.
- **Custom workload and failure analysis:** custom Python pipeline (normalization, grading, comparison, case studies).
- **Performance:** AIPerf (plus request-log fallback summaries).
- **Endpoints:** Hugging Face endpoints preferred, with manual URL fallback.
- **Execution policy:** sequential backend runs for cost safety (`RUN_BACKENDS_SEQUENTIALLY={str(config.run_backends_sequentially).lower()}`).

## 7) Tradeoffs and limitations
- Managed endpoints provide limited low-level scheduler/KV/GPU observability.
- Some long-context public tasks are expensive; the default run uses a bounded slice.
- AIPerf synthetic profiles are useful for stress tests but are not substitutes for task-level correctness analysis.
- Endpoint lifecycle and quotas can still affect runtime and cold-start behavior.
"""

    memo_path.write_text(content, encoding="utf-8")
    return memo_path
