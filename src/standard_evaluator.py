from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .utils import append_jsonl, ensure_dir


def _should_run(config: AppConfig, optimization_mode: str, repeat_index: int) -> bool:
    if not config.standard_eval_enabled or config.standard_evaluator == "none":
        return False
    if config.standard_eval_run_on_all_conditions:
        return True
    return optimization_mode == "baseline" and repeat_index == 1


def _has_humaneval(tasks: list[str]) -> bool:
    return any("humaneval" in str(t).lower() for t in tasks)


def _extract_code_completion(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""

    fenced_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", value, flags=re.IGNORECASE)
    if fenced_blocks:
        preferred = next((b for b in fenced_blocks if "def " in b or "async def " in b), fenced_blocks[0])
        value = preferred.strip()
    else:
        start_match = re.search(r"(?m)^(?:\s*)(?:def |async def |from |import )", value)
        if start_match:
            value = value[start_match.start() :].strip()

    # Drop stray markdown fence markers and trailing prose.
    value = value.replace("```python", "").replace("```", "").strip()
    cut_match = re.search(r"(?m)^\s*(?:Explanation:|Here'?s|This function|The function)", value)
    if cut_match and cut_match.start() > 0:
        value = value[: cut_match.start()].rstrip()
    return value


def _normalize_humaneval_samples(
    out_dir: Path,
    logs: Path,
    backend: str,
    optimization_mode: str,
    repeat_index: int,
) -> None:
    sample_files = sorted(out_dir.glob("**/samples_humaneval_instruct_*.jsonl"))
    if not sample_files:
        append_jsonl(
            logs,
            {
                "event": "lm_eval_humaneval_normalization_skipped",
                "backend": backend,
                "optimization_mode": optimization_mode,
                "repeat_index": repeat_index,
                "reason": "no_humaneval_samples_found",
            },
        )
        return

    manifest_rows: list[dict[str, object]] = []
    for sample_file in sample_files:
        raw_lines = sample_file.read_text(encoding="utf-8").splitlines()
        normalized_lines: list[str] = []
        modified = 0

        for line in raw_lines:
            row = line.strip()
            if not row:
                continue
            data = json.loads(row)
            resps = data.get("resps")
            original = ""
            normalized = ""

            if isinstance(resps, list) and resps:
                first = resps[0]
                if isinstance(first, list) and first and isinstance(first[0], str):
                    original = first[0]
                    normalized = _extract_code_completion(original)
                    if normalized and normalized != original:
                        first[0] = normalized
                        modified += 1
                elif isinstance(first, str):
                    original = first
                    normalized = _extract_code_completion(original)
                    if normalized and normalized != original:
                        resps[0] = normalized
                        modified += 1

            normalized_lines.append(json.dumps(data, ensure_ascii=False))

        normalized_path = sample_file.with_name(sample_file.stem + "_normalized.jsonl")
        normalized_path.write_text("\n".join(normalized_lines) + ("\n" if normalized_lines else ""), encoding="utf-8")
        manifest_rows.append(
            {
                "source_file": str(sample_file),
                "normalized_file": str(normalized_path),
                "rows": len(normalized_lines),
                "rows_modified": modified,
            }
        )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "optimization_mode": optimization_mode,
        "repeat_index": repeat_index,
        "entries": manifest_rows,
    }
    manifest_path = out_dir / "humaneval_normalization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_jsonl(
        logs,
        {
            "event": "lm_eval_humaneval_samples_normalized",
            "backend": backend,
            "optimization_mode": optimization_mode,
            "repeat_index": repeat_index,
            "manifest_path": str(manifest_path),
            "files_processed": len(manifest_rows),
            "rows_modified": sum(int(x["rows_modified"]) for x in manifest_rows),
        },
    )


def run_standard_evaluator(
    config: AppConfig,
    run_dir: Path,
    backend: str,
    endpoint_url: str,
    optimization_mode: str,
    repeat_index: int,
    smoke: bool,
) -> Path | None:
    """
    Runs a standard evaluator (currently lm-eval-harness) for public benchmark checks.
    This is additive: custom pipeline still drives per-sample bucket analysis.
    """
    logs = run_dir / "logs" / "standard_evaluator.jsonl"

    if not _should_run(config, optimization_mode, repeat_index):
        append_jsonl(
            logs,
            {
                "event": "standard_eval_skipped_condition",
                "backend": backend,
                "optimization_mode": optimization_mode,
                "repeat_index": repeat_index,
            },
        )
        return None

    if config.standard_evaluator != "lm_eval":
        append_jsonl(logs, {"event": "standard_eval_unknown_evaluator", "value": config.standard_evaluator})
        return None

    if shutil.which("lm_eval") is None:
        append_jsonl(logs, {"event": "lm_eval_not_installed", "backend": backend})
        return None

    out_dir = (
        run_dir
        / "raw"
        / "standard_evals"
        / "lm_eval"
        / backend
        / optimization_mode
        / f"repeat_{repeat_index}"
    )
    ensure_dir(out_dir)

    limit = config.smoke_limit if smoke else config.limit
    if config.standard_eval_limit_override > 0:
        limit = min(limit, config.standard_eval_limit_override)

    base_url = endpoint_url.rstrip("/") + "/v1/chat/completions"
    task_str = ",".join(config.standard_eval_tasks)
    model_id = config.model_id_for_backend(backend)
    model_args = f"model={model_id},base_url={base_url}"
    if _has_humaneval(config.standard_eval_tasks) and config.standard_eval_humaneval_code_only_prompt:
        safe_system_prompt = config.standard_eval_humaneval_system_prompt.replace(",", " ")
        model_args += f",system={safe_system_prompt}"
    cmd = [
        "lm_eval",
        "run",
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        task_str,
        "--limit",
        str(limit),
        "--apply_chat_template",
        "--log_samples",
        "--output_path",
        str(out_dir),
    ]
    if _has_humaneval(config.standard_eval_tasks):
        cmd.append("--confirm_run_unsafe_code")

    append_jsonl(
        logs,
        {
            "event": "lm_eval_start",
            "backend": backend,
            "optimization_mode": optimization_mode,
            "repeat_index": repeat_index,
            "tasks": config.standard_eval_tasks,
            "limit": limit,
            "output_path": str(out_dir),
        },
    )
    try:
        env = os.environ.copy()
        if config.hf_token:
            # local-chat-completions reads OPENAI_API_KEY from env.
            env["OPENAI_API_KEY"] = config.hf_token
            # Ensure HF Hub-backed task assets can authenticate when needed.
            env["HF_TOKEN"] = config.hf_token
            env["HUGGINGFACEHUB_API_TOKEN"] = config.hf_token
        # Humaneval relies on execute-on-generated-code metric; lm-eval requires explicit opt-in.
        if _has_humaneval(config.standard_eval_tasks):
            env["HF_ALLOW_CODE_EVAL"] = "1"
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        append_jsonl(
            logs,
            {
                "event": "lm_eval_stdout_preview",
                "backend": backend,
                "preview": (result.stdout or "")[:1000],
            },
        )
        append_jsonl(
            logs,
            {
                "event": "lm_eval_completed",
                "backend": backend,
                "optimization_mode": optimization_mode,
                "repeat_index": repeat_index,
                "tasks": config.standard_eval_tasks,
                "output_path": str(out_dir),
            },
        )
        if _has_humaneval(config.standard_eval_tasks):
            _normalize_humaneval_samples(
                out_dir=out_dir,
                logs=logs,
                backend=backend,
                optimization_mode=optimization_mode,
                repeat_index=repeat_index,
            )
    except Exception as exc:  # noqa: BLE001
        stderr_preview = ""
        stdout_preview = ""
        if isinstance(exc, subprocess.CalledProcessError):
            stderr_preview = (exc.stderr or "")[:20000]
            stdout_preview = (exc.stdout or "")[:20000]
        append_jsonl(
            logs,
            {
                "event": "lm_eval_failed",
                "backend": backend,
                "optimization_mode": optimization_mode,
                "repeat_index": repeat_index,
                "error": str(exc),
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
            },
        )
    return out_dir
