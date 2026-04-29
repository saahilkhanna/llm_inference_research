from __future__ import annotations

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, RepoFile

from .config import AppConfig
from .utils import append_jsonl
from .utils import write_json


def _redact_endpoint_payload_for_logs(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    secrets = out.get("model", {}).get("secrets")
    if isinstance(secrets, dict):
        for k in secrets:
            if secrets[k]:
                secrets[k] = "[REDACTED]"
    return out


@dataclass
class EndpointInfo:
    key: str
    backend: str
    optimization_mode: str
    name: str
    url: str
    created: bool
    model_id: str
    endpoint_args: list[str]


def endpoint_key(backend: str, optimization_mode: str) -> str:
    return f"{backend}:{optimization_mode}"


def _endpoint_name(backend: str, optimization_mode: str, run_id: str) -> str:
    mode_tag = {
        "baseline": "base",
        "kv_cache_quant": "kvq",
        "spec_decode": "spec",
    }.get(optimization_mode, optimization_mode)
    base = f"{backend}-{mode_tag}-{run_id}".lower().replace("_", "-")
    # HF endpoint names must be lowercase alnum/hyphen and start/end alnum.
    sanitized = re.sub(r"[^a-z0-9-]+", "-", base)
    truncated = sanitized[:50].strip("-")
    if len(truncated) < 4:
        truncated = f"{backend[:3]}-{mode_tag}"[:50].strip("-")
    return truncated


def _manual_endpoint_url(config: AppConfig, backend: str, optimization_mode: str) -> str:
    if backend == "vllm":
        by_mode = {
            "baseline": config.vllm_endpoint_url_baseline,
            "kv_cache_quant": config.vllm_endpoint_url_kv_cache_quant,
            "spec_decode": config.vllm_endpoint_url_spec_decode,
        }
        return by_mode.get(optimization_mode, "") or config.vllm_endpoint_url
    if backend == "sglang":
        by_mode = {
            "baseline": config.sglang_endpoint_url_baseline,
            "kv_cache_quant": config.sglang_endpoint_url_kv_cache_quant,
            "spec_decode": config.sglang_endpoint_url_spec_decode,
        }
        return by_mode.get(optimization_mode, "") or config.sglang_endpoint_url
    if backend == "llama_cpp":
        return config.llama_cpp_endpoint_url if optimization_mode == "baseline" else ""
    raise ValueError(f"Unknown backend: {backend}")


def endpoint_for_condition(endpoints: dict[str, EndpointInfo], backend: str, optimization_mode: str) -> EndpointInfo:
    key = endpoint_key(backend, optimization_mode)
    if key not in endpoints:
        raise KeyError(f"No endpoint resolved for {key}")
    return endpoints[key]


def _endpoint_model_id(config: AppConfig, backend: str) -> str:
    if backend == "llama_cpp":
        return config.llama_cpp_model_id
    if backend == "vllm":
        return config.vllm_model_id
    if backend == "sglang":
        return config.sglang_model_id
    raise ValueError(f"Unknown backend: {backend}")


def _endpoint_framework(config: AppConfig, backend: str) -> str:
    if backend == "llama_cpp":
        return config.llama_cpp_endpoint_framework
    if backend == "vllm":
        return config.vllm_endpoint_framework
    if backend == "sglang":
        return config.sglang_endpoint_framework
    raise ValueError(f"Unknown backend: {backend}")


def _api_base(config: AppConfig) -> str:
    if not config.hf_namespace:
        raise ValueError("HF_NAMESPACE is required when CREATE_ENDPOINTS=true.")
    return f"https://api.endpoints.huggingface.co/v2/endpoint/{config.hf_namespace}"


def _headers(config: AppConfig) -> dict[str, str]:
    if not config.hf_token:
        raise ValueError("HF_TOKEN is required when CREATE_ENDPOINTS=true.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.hf_token}",
    }


def _engine_image(config: AppConfig, backend: str, llamacpp_model_path: str | None = None) -> dict[str, Any]:
    if backend == "vllm":
        return {
            "vLLM": {
                "tensorParallelSize": config.endpoint_tensor_parallel_size,
                "healthRoute": config.vllm_engine_health_route,
                "port": config.vllm_engine_port,
                "url": config.vllm_engine_image_url,
            }
        }
    if backend == "sglang":
        return {
            "sGLang": {
                "tensorParallelSize": config.endpoint_tensor_parallel_size,
                "healthRoute": config.sglang_engine_health_route,
                "port": config.sglang_engine_port,
                "url": config.sglang_engine_image_url,
            }
        }
    if backend == "llama_cpp":
        if not llamacpp_model_path:
            raise ValueError("llamacpp_model_path is required for llama_cpp engine image")
        llama: dict[str, Any] = {
            "url": config.llama_cpp_engine_image_url,
            "healthRoute": config.llama_cpp_engine_health_route,
            "port": config.llama_cpp_engine_port,
            "modelPath": llamacpp_model_path,
            "ctxSize": config.llama_cpp_ctx_size,
            "nParallel": config.llama_cpp_n_parallel,
            "threadsHttp": config.llama_cpp_threads_http,
        }
        if config.llama_cpp_n_gpu_layers is not None:
            llama["nGpuLayers"] = config.llama_cpp_n_gpu_layers
        if config.llama_cpp_gguf_variant is not None:
            llama["variant"] = config.llama_cpp_gguf_variant
        return {"llamacpp": llama}
    return {"huggingface": {}}


def _pick_gguf_from_repo(repo_id: str, token: str) -> str:
    api = HfApi(token=token)
    gguf_files = sorted(
        e.path
        for e in api.list_repo_tree(repo_id, repo_type="model", recursive=False)
        if isinstance(e, RepoFile) and e.path.endswith(".gguf")
    )
    if not gguf_files:
        raise RuntimeError(f"No .gguf files found at repo root for {repo_id}")
    for needle in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"):
        for name in gguf_files:
            if needle in name:
                return name
    return gguf_files[0]


def _llamacpp_model_path(config: AppConfig, logs_path: Path) -> str:
    if config.llama_cpp_gguf_filename:
        return config.llama_cpp_gguf_filename
    if not config.hf_token:
        raise ValueError("Set LLAMA_CPP_GGUF_FILENAME or provide HF_TOKEN to auto-select a .gguf file.")
    try:
        choice = _pick_gguf_from_repo(config.llama_cpp_model_id, config.hf_token)
    except Exception as error:
        raise RuntimeError(
            "Failed listing GGUF files from the Hub while auto-selecting LLAMA_CPP_GGUF_FILENAME. "
            "Set LLAMA_CPP_GGUF_FILENAME explicitly (recommended for reproducibility), "
            "or fix TLS/network access to huggingface.co. "
            f"Underlying error: {error}"
        ) from error
    append_jsonl(
        logs_path,
        {
            "event": "llamacpp_gguf_auto",
            "repository": config.llama_cpp_model_id,
            "modelPath": choice,
        },
    )
    return choice


def _endpoint_payload(
    config: AppConfig,
    backend: str,
    optimization_mode: str,
    name: str,
    *,
    llamacpp_model_path: str | None = None,
) -> dict[str, Any]:
    endpoint_args = config.endpoint_args_for(backend, optimization_mode)
    model: dict[str, Any] = {
        "env": {},
        "framework": _endpoint_framework(config, backend),
        "image": _engine_image(config, backend, llamacpp_model_path=llamacpp_model_path),
        "repository": _endpoint_model_id(config, backend),
        "secrets": {},
        "task": "text-generation",
    }
    # vLLM/SGLang may download extra Hub repos (e.g. speculative draft) — gated models need HF_TOKEN in-container.
    if config.hf_token:
        model["secrets"]["HF_TOKEN"] = config.hf_token
    # HF API rejects empty args; omit the field when there are no CLI args.
    if endpoint_args:
        model["args"] = endpoint_args

    return {
        "cacheHttpResponses": config.endpoint_cache_http_responses,
        "compute": {
            "accelerator": config.endpoint_accelerator,
            "instanceSize": config.endpoint_instance_size,
            "instanceType": config.endpoint_instance_type,
            "scaling": {
                "maxReplica": config.endpoint_max_replica,
                "measure": {"hardwareUsage": config.endpoint_hardware_usage_target},
                "metric": "hardwareUsage",
                "minReplica": config.endpoint_min_replica,
                "scaleToZeroTimeout": config.endpoint_scale_to_zero_timeout,
            },
        },
        "model": model,
        "name": name,
        "notifications": {
            "email": config.endpoint_notifications_email,
            "push": config.endpoint_notifications_push,
        },
        "provider": {
            "region": config.endpoint_region,
            "vendor": config.endpoint_vendor,
        },
        "tags": [],
        "type": config.endpoint_type,
    }


def _request_json(config: AppConfig, method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    response = requests.request(method, url, headers=_headers(config), json=payload, timeout=120)
    try:
        data = response.json() if response.text else {}
    except ValueError:
        data = {"raw": response.text}
    return response.status_code, data


def _delete_endpoint(config: AppConfig, endpoint_name: str, logs_path: Path) -> None:
    status, data = _request_json(config, "DELETE", f"{_api_base(config)}/{endpoint_name}")
    append_jsonl(logs_path, {"event": "endpoint_delete", "name": endpoint_name, "status": status, "response": data})
    if status not in {200, 202, 204, 404}:
        raise RuntimeError(f"Failed to delete existing endpoint {endpoint_name}: HTTP {status} - {data}")


def _create_or_replace_endpoint(config: AppConfig, payload: dict[str, Any], logs_path: Path) -> None:
    name = payload["name"]
    status, data = _request_json(config, "POST", _api_base(config), payload)
    append_jsonl(logs_path, {"event": "endpoint_create_response", "name": name, "status": status, "response": data})
    if status in {200, 201}:
        return
    if status == 409:
        _delete_endpoint(config, name, logs_path)
        status, data = _request_json(config, "POST", _api_base(config), payload)
        append_jsonl(logs_path, {"event": "endpoint_recreate_response", "name": name, "status": status, "response": data})
        if status in {200, 201}:
            return
    if status == 403:
        raise RuntimeError(
            "Failed to create endpoint due to authorization or gated model access. "
            f"Server response: {data}"
        )
    raise RuntimeError(f"Failed to create endpoint {name}: HTTP {status} - {data}")


def _wait_for_ready(config: AppConfig, endpoint_name: str, logs_path: Path, timeout_seconds: int = 1800) -> str:
    deadline = time.monotonic() + timeout_seconds
    poll_seconds = 15
    url = f"{_api_base(config)}/{endpoint_name}"
    while time.monotonic() < deadline:
        status_code, data = _request_json(config, "GET", url)
        # Parallel creates sometimes return 404 briefly until the resource is visible (HF API propagation).
        if status_code == 404:
            append_jsonl(
                logs_path,
                {"event": "endpoint_poll_not_visible_yet", "name": endpoint_name, "status_code": status_code},
            )
            time.sleep(poll_seconds)
            continue
        if status_code >= 400:
            raise RuntimeError(f"Failed polling {endpoint_name}: HTTP {status_code} - {data}")

        status = data.get("status", {}).get("state", "unknown")
        endpoint_url = data.get("status", {}).get("url")
        append_jsonl(
            logs_path,
            {"event": "endpoint_poll", "name": endpoint_name, "status": status, "url": endpoint_url},
        )
        if status == "running" and endpoint_url:
            return endpoint_url
        if status in {"failed", "updateFailed"}:
            raise RuntimeError(f"{endpoint_name} failed while starting: {data.get('status', {})}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for {endpoint_name}")


def create_endpoint_if_needed(
    config: AppConfig, backend: str, optimization_mode: str, run_id: str, logs_path: Path
) -> EndpointInfo:
    key = endpoint_key(backend, optimization_mode)
    existing_url = _manual_endpoint_url(config, backend, optimization_mode)
    if existing_url:
        return EndpointInfo(
            key=key,
            backend=backend,
            optimization_mode=optimization_mode,
            name=f"manual-{backend}-{optimization_mode}",
            url=existing_url,
            created=False,
            model_id=_endpoint_model_id(config, backend),
            endpoint_args=config.endpoint_args_for(backend, optimization_mode),
        )

    if not config.create_endpoints:
        raise ValueError(f"{key} endpoint URL missing and CREATE_ENDPOINTS=false")

    name = _endpoint_name(backend, optimization_mode, run_id)
    append_jsonl(
        logs_path,
        {
            "event": "endpoint_create_start",
            "key": key,
            "backend": backend,
            "optimization_mode": optimization_mode,
            "name": name,
        },
    )

    llamacpp_model_path = None
    if backend == "llama_cpp":
        llamacpp_model_path = _llamacpp_model_path(config, logs_path)

    payload = _endpoint_payload(
        config, backend, optimization_mode, name, llamacpp_model_path=llamacpp_model_path
    )
    append_jsonl(
        logs_path,
        {
            "event": "endpoint_payload",
            "key": key,
            "payload": _redact_endpoint_payload_for_logs(payload),
        },
    )
    _create_or_replace_endpoint(config, payload, logs_path)
    endpoint_url = _wait_for_ready(config, name, logs_path, timeout_seconds=config.endpoint_ready_timeout_seconds)
    append_jsonl(
        logs_path,
        {
            "event": "endpoint_create_ready",
            "key": key,
            "backend": backend,
            "optimization_mode": optimization_mode,
            "name": name,
            "repository": payload["model"]["repository"],
            **(
                {"llamacppModelPath": (payload["model"].get("image") or {}).get("llamacpp", {}).get("modelPath")}
                if backend == "llama_cpp"
                else {}
            ),
            **({"args": payload["model"]["args"]} if "args" in payload["model"] else {}),
        },
    )
    return EndpointInfo(
        key=key,
        backend=backend,
        optimization_mode=optimization_mode,
        name=name,
        url=endpoint_url,
        created=True,
        model_id=payload["model"]["repository"],
        endpoint_args=payload["model"].get("args", []),
    )


def endpoint_modes_for_engine(backend: str, optimization_modes: list[str]) -> list[str]:
    if backend == "llama_cpp":
        return ["baseline"]
    return optimization_modes


def resolve_endpoints(
    config: AppConfig, run_id: str, run_dir: Path, optimization_modes: list[str] | None = None
) -> dict[str, EndpointInfo]:
    logs_path = run_dir / "logs" / "endpoint_manager.jsonl"
    modes = optimization_modes or config.optimization_modes
    tasks: list[tuple[str, str]] = [
        (backend, optimization_mode)
        for backend in config.engines
        for optimization_mode in endpoint_modes_for_engine(backend, modes)
    ]
    endpoints: dict[str, EndpointInfo] = {}
    if not tasks:
        return endpoints

    def _create(backend: str, optimization_mode: str) -> EndpointInfo:
        return create_endpoint_if_needed(config, backend, optimization_mode, run_id, logs_path)

    if config.provision_endpoints_parallel:
        max_workers = min(len(tasks), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_create, b, om) for b, om in tasks]
            for fut in as_completed(futures):
                info = fut.result()
                endpoints[info.key] = info
    else:
        for backend, optimization_mode in tasks:
            info = _create(backend, optimization_mode)
            endpoints[info.key] = info

    verification = {
        "total_conditions": len(tasks),
        "parallel_provisioning_enabled": bool(config.provision_endpoints_parallel),
        "conditions": [
            {
                "key": key,
                "backend": info.backend,
                "optimization_mode": info.optimization_mode,
                "name": info.name,
                "url": info.url,
                "created": info.created,
                "model_id": info.model_id,
                "endpoint_args": info.endpoint_args,
            }
            for key, info in sorted(endpoints.items(), key=lambda item: item[0])
        ],
    }
    write_json(run_dir / "logs" / "endpoint_resolution.json", verification)
    append_jsonl(logs_path, {"event": "endpoint_resolution_written", "path": str(run_dir / "logs" / "endpoint_resolution.json")})
    return endpoints


def apply_shutdown_mode(config: AppConfig, endpoints: dict[str, Any], run_dir: Path) -> None:
    if config.shutdown_mode == "none":
        return

    from huggingface_hub import HfApi  # type: ignore
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore

    api = HfApi(token=config.hf_token) if config.hf_token else HfApi()
    logs_path = run_dir / "logs" / "endpoint_shutdown.jsonl"

    for key, info in endpoints.items():
        if not info:
            continue
        name = info.get("name") if isinstance(info, dict) else getattr(info, "name", "")
        created = info.get("created") if isinstance(info, dict) else getattr(info, "created", False)
        if not created or not name or name.startswith("manual-"):
            append_jsonl(logs_path, {"event": "skip_shutdown_manual", "key": key, "name": name})
            continue
        try:
            endpoint = api.get_inference_endpoint(name=name, namespace=config.hf_namespace or None)
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 404:
                append_jsonl(
                    logs_path,
                    {"event": "skip_shutdown_endpoint_missing", "key": key, "name": name, "detail": str(exc)},
                )
                continue
            raise
        if config.shutdown_mode == "pause":
            endpoint.pause()
        elif config.shutdown_mode == "scale_to_zero":
            endpoint.scale_to_zero()
        elif config.shutdown_mode == "delete":
            endpoint.delete()
        append_jsonl(
            logs_path,
            {"event": "shutdown_applied", "key": key, "name": name, "mode": config.shutdown_mode},
        )
