from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .normalize_outputs import extract_numeric
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
) -> tuple[str, dict[str, Any], dict[str, Any]]:
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
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_params:
        payload.update(extra_params)
    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=config.request_timeout_seconds, stream=True)
        response.raise_for_status()
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        ttft_seconds: float | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_line = line.removeprefix("data:").strip()
            if data_line == "[DONE]":
                break
            event = json.loads(data_line)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices", [])
            if not choices:
                continue
            choice0 = choices[0] or {}
            delta = choice0.get("delta", {}) if isinstance(choice0.get("delta", {}), dict) else {}
            token_piece = delta.get("content", "")
            if token_piece:
                if ttft_seconds is None:
                    ttft_seconds = max(0.0, time.perf_counter() - started)
                chunks.append(str(token_piece))
        text = "".join(chunks)
        raw = {"choices": [{"message": {"content": text}}], "usage": usage, "_streaming": True}
        return text or "", raw, {"ttft_seconds": ttft_seconds}
    except Exception:
        # Fallback to non-streaming for endpoints that do not support stream semantics.
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        response = requests.post(url, json=payload, headers=headers, timeout=config.request_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text or "", data, {"ttft_seconds": None}


def _call_tgi(
    endpoint_url: str,
    config: AppConfig,
    prompt: str,
    token: str,
    model_id: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
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
    return text or "", data, {"ttft_seconds": None}


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


_NUM_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def call_backend(
    endpoint_url: str, config: AppConfig, prompt: str, optimization_mode: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    return call_backend_for_engine(endpoint_url, config, prompt, optimization_mode, "unknown")


def call_backend_for_engine(
    endpoint_url: str, config: AppConfig, prompt: str, optimization_mode: str, backend: str
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    style = config.backend_api_style
    errors: list[str] = []
    hints = _optimization_hints(config, optimization_mode)
    model_id = config.model_id_for_backend(backend)

    if style in {"auto", "openai_chat"}:
        try:
            text, raw, perf = _call_openai_chat(endpoint_url, config, prompt, config.hf_token, model_id, hints)
            raw["_optimization_hints"] = hints
            return text, raw, perf
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openai_chat: {exc}")
            if style == "openai_chat":
                raise
    if style in {"auto", "tgi"}:
        try:
            text, raw, perf = _call_tgi(endpoint_url, config, prompt, config.hf_token, model_id, hints)
            raw["_optimization_hints"] = hints
            return text, raw, perf
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
        perf_metrics: dict[str, Any] = {}

        try:
            response_text, raw_response, perf_metrics = call_backend_for_engine(
                endpoint_url, config, sample["prompt"], optimization_mode, backend
            )
        except Exception as exc:  # noqa: BLE001
            success = False
            error_message = str(exc)

        end = time.perf_counter()
        end_iso = _now_iso()
        latency = max(0.0, end - start)
        usage = raw_response.get("usage", {}) if isinstance(raw_response.get("usage", {}), dict) else {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        prompt_tokens_est = len(str(sample["prompt"] or "").split())
        completion_tokens_est = len(str(response_text or "").split())
        ttft_seconds = perf_metrics.get("ttft_seconds")
        tpot_seconds = None
        if completion_tokens and ttft_seconds is not None:
            decode_time = max(0.0, latency - float(ttft_seconds))
            tpot_seconds = decode_time / max(1, int(completion_tokens))
        tokens_per_second = None
        if completion_tokens:
            tokens_per_second = float(completion_tokens) / max(latency, 1e-9)
        in_cost_1k = float(os.getenv("COST_PER_1K_INPUT_TOKENS_USD", "0"))
        out_cost_1k = float(os.getenv("COST_PER_1K_OUTPUT_TOKENS_USD", "0"))
        estimated_cost_usd = None
        if prompt_tokens is not None and completion_tokens is not None and (in_cost_1k > 0 or out_cost_1k > 0):
            estimated_cost_usd = (float(prompt_tokens) / 1000.0) * in_cost_1k + (
                float(completion_tokens) / 1000.0
            ) * out_cost_1k
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_est": prompt_tokens_est,
            "completion_tokens_est": completion_tokens_est,
            "token_counts_missing": prompt_tokens is None or completion_tokens is None,
            "ttft_seconds": ttft_seconds,
            "ttft_missing": ttft_seconds is None,
            "tpot_seconds": tpot_seconds,
            "tokens_per_second": tokens_per_second,
            "estimated_cost_usd": estimated_cost_usd,
            "prompt_char_len": len(sample["prompt"] or ""),
            "prompt_line_count": str(sample["prompt"] or "").count("\n") + 1,
            "prompt_numeric_token_count": len(_NUM_TOKEN_RE.findall(str(sample["prompt"] or ""))),
            "response_char_len": len(response_text or ""),
            "response_line_count": str(response_text or "").count("\n") + (1 if response_text else 0),
            "response_has_answer_tag": bool(re.search(r"(?i)\banswer\s*:", str(response_text or ""))),
            "response_numeric_extracted": extract_numeric(str(response_text or "")) if response_text else None,
            "expected_numeric_extracted": extract_numeric(str(sample.get("expected_answer", ""))),
            "condition_fingerprint": f"{backend}|{optimization_mode}|{repeat_index}",
            "endpoint_fingerprint": hashlib.sha256(endpoint_url.encode("utf-8")).hexdigest()[:16],
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
