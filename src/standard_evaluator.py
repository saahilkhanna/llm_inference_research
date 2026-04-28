from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import AppConfig
from .utils import append_jsonl, ensure_dir


def _should_run(config: AppConfig, optimization_mode: str, repeat_index: int) -> bool:
    if not config.standard_eval_enabled or config.standard_evaluator == "none":
        return False
    if config.standard_eval_run_on_all_conditions:
        return True
    return optimization_mode == "baseline" and repeat_index == 1


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
    cmd = [
        "lm_eval",
        "run",
        "--model",
        "local-chat-completions",
        "--model_args",
        f"model={config.model_id},base_url={base_url}",
        "--tasks",
        task_str,
        "--limit",
        str(limit),
        "--apply_chat_template",
        "--log_samples",
        "--output_path",
        str(out_dir),
    ]

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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        append_jsonl(
            logs,
            {
                "event": "lm_eval_stdout_preview",
                "backend": backend,
                "preview": (result.stdout or "")[:1000],
            },
        )
        append_jsonl(logs, {"event": "lm_eval_completed", "backend": backend, "output_path": str(out_dir)})
    except Exception as exc:  # noqa: BLE001
        stderr_preview = ""
        stdout_preview = ""
        if isinstance(exc, subprocess.CalledProcessError):
            stderr_preview = (exc.stderr or "")[:2000]
            stdout_preview = (exc.stdout or "")[:2000]
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
