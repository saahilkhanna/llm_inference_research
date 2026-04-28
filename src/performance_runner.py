from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import AppConfig
from .utils import append_jsonl, ensure_dir


def run_aiperf_profile(
    backend: str,
    endpoint_url: str,
    config: AppConfig,
    run_dir: Path,
    smoke: bool,
) -> Path | None:
    logs_path = run_dir / "logs" / "aiperf.jsonl"
    if not config.aiperf_enabled:
        append_jsonl(logs_path, {"event": "aiperf_disabled", "backend": backend})
        return None
    if shutil.which("aiperf") is None:
        append_jsonl(logs_path, {"event": "aiperf_not_found", "backend": backend})
        return None

    request_count = 6 if smoke else 20
    artifact_dir = run_dir / "raw" / "aiperf" / backend
    ensure_dir(artifact_dir)

    # Synthetic profile mode is optional and scoped for performance-only analysis.
    cmd = [
        "aiperf",
        "profile",
        "--model",
        config.model_id,
        "--url",
        endpoint_url,
        "--endpoint-type",
        "chat",
        "--request-count",
        str(request_count),
        "--concurrency",
        "1",
        "--artifact-dir",
        str(artifact_dir),
        "--export-level",
        "records",
        "--ui",
        "none",
    ]
    if config.aiperf_synthetic_enabled:
        cmd.extend(["--synthetic-input-tokens-mean", "256", "--output-tokens-mean", "64"])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        append_jsonl(logs_path, {"event": "aiperf_completed", "backend": backend, "artifact_dir": str(artifact_dir)})
        return artifact_dir
    except Exception as exc:  # noqa: BLE001
        append_jsonl(logs_path, {"event": "aiperf_failed", "backend": backend, "error": str(exc)})
        return None
