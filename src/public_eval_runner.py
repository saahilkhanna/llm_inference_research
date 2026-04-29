from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .utils import append_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _openai_chat_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/chat/completions"
    if stripped.endswith("/v1/"):
        return f"{stripped}chat/completions"
    return f"{stripped}/v1/chat/completions"


def _call_openai_chat(
    endpoint_url: str,
    config: AppConfig,
    prompt: str,
    token: str,
    model_id: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    url = _openai_chat_url(endpoint_url)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }
    if extra_params:
        payload.update(extra_params)
    response = requests.post(url, json=payload, headers=headers, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text or "", data


def _call_tgi(
    endpoint_url: str,
    config: AppConfig,
    prompt: str,
    token: str,
    model_id: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
        },
    }
    if extra_params:
        payload["parameters"].update(extra_params)
    response = requests.post(endpoint_url, json=payload, headers=headers, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        text = data[0].get("generated_text", "")
    else:
        text = data.get("generated_text") or data.get("text", "")
    return text or "", data


def _optimization_hints(config: AppConfig, optimization_mode: str) -> dict[str, Any]:
    if not config.apply_optimization_request_hints:
        return {}
    if optimization_mode == "kv_cache_quant":
        return {"kv_cache_dtype": config.kv_cache_quant_mode}
    if optimization_mode == "spec_decode":
        hints: dict[str, Any] = {"speculative_decoding": True, "num_speculative_tokens": config.speculative_num_tokens}
        if config.speculative_draft_model:
            hints["draft_model"] = config.speculative_draft_model
        return hints
    return {}


def call_backend(endpoint_url: str, config: AppConfig, prompt: str, optimization_mode: str) -> tuple[str, dict[str, Any]]:
    return call_backend_for_engine(endpoint_url, config, prompt, optimization_mode, "unknown")


def call_backend_for_engine(
    endpoint_url: str, config: AppConfig, prompt: str, optimization_mode: str, backend: str
) -> tuple[str, dict[str, Any]]:
    style = config.backend_api_style
    errors: list[str] = []
    hints = _optimization_hints(config, optimization_mode)
    model_id = config.model_id_for_backend(backend)

    if style in {"auto", "openai_chat"}:
        try:
            text, raw = _call_openai_chat(endpoint_url, config, prompt, config.hf_token, model_id, hints)
            raw["_optimization_hints"] = hints
            return text, raw
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openai_chat: {exc}")
            if style == "openai_chat":
                raise
    if style in {"auto", "tgi"}:
        try:
            text, raw = _call_tgi(endpoint_url, config, prompt, config.hf_token, model_id, hints)
            raw["_optimization_hints"] = hints
            return text, raw
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tgi: {exc}")
            raise RuntimeError("; ".join(errors)) from exc
    raise RuntimeError(f"Unsupported BACKEND_API_STYLE: {style}")


def run_samples_for_backend(
    backend: str,
    endpoint_url: str,
    samples: list[dict[str, Any]],
    config: AppConfig,
    run_dir: Path,
    optimization_mode: str = "baseline",
    repeat_index: int = 1,
) -> Path:
    out_path = run_dir / "raw" / f"responses_{backend}_{optimization_mode}_repeat_{repeat_index}.jsonl"
    progress_log = run_dir / "logs" / "run_progress.jsonl"
    total = len(samples)
    started = time.perf_counter()
    checkpoint_every = max(1, total // 10)
    for idx, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        start_iso = _now_iso()
        success = True
        error_message = ""
        response_text = ""
        raw_response: dict[str, Any] = {}

        try:
            response_text, raw_response = call_backend_for_engine(
                endpoint_url, config, sample["prompt"], optimization_mode, backend
            )
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)

        end = time.perf_counter()
        end_iso = _now_iso()
        latency = max(0.0, end - start)
        record = {
            "sample_id": sample["sample_id"],
            "workload": sample["workload"],
            "task_id": sample["task_id"],
            "workload_class": sample.get("workload_class", "unknown"),
            "prompt": sample["prompt"],
            "expected_answer": sample.get("expected_answer"),
            "grading_type": sample.get("grading_type", "unknown"),
            "backend": backend,
            "engine": backend,
            "optimization_mode": optimization_mode,
            "repeat_index": repeat_index,
            "response_text": response_text,
            "raw_response_json": raw_response,
            "success": success,
            "error_message": error_message,
            "start_time": start_iso,
            "end_time": end_iso,
            "latency_seconds": latency,
            "decoding_parameters": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
                "model_id": config.model_id_for_backend(backend),
            },
            "token_counts": raw_response.get("usage", {}),
        }
        append_jsonl(out_path, record)
        if idx == 1 or idx == total or idx % checkpoint_every == 0:
            elapsed = max(0.0, time.perf_counter() - started)
            avg_per_sample = elapsed / idx if idx else 0.0
            remaining = max(0.0, (total - idx) * avg_per_sample)
            append_jsonl(
                progress_log,
                {
                    "event": "samples_progress",
                    "backend": backend,
                    "optimization_mode": optimization_mode,
                    "repeat_index": repeat_index,
                    "completed_samples": idx,
                    "total_samples": total,
                    "percent_complete": round((idx / total) * 100.0, 2) if total else 100.0,
                    "elapsed_seconds": elapsed,
                    "eta_remaining_seconds": remaining,
                },
            )
    return out_path
