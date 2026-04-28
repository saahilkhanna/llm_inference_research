from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import append_jsonl


@dataclass
class EndpointInfo:
    name: str
    url: str
    created: bool


def _endpoint_name(config: AppConfig, backend: str, run_id: str) -> str:
    base = f"{backend}-{run_id}".lower().replace("_", "-")
    return base[:50]


def create_endpoint_if_needed(
    api: HfApi, config: AppConfig, backend: str, run_id: str, logs_path: Path
) -> EndpointInfo:
    if backend == "vllm":
        existing_url = config.vllm_endpoint_url
    elif backend == "sglang":
        existing_url = config.sglang_endpoint_url
    elif backend == "llama_cpp":
        existing_url = config.llama_cpp_endpoint_url
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if existing_url:
        return EndpointInfo(name=f"manual-{backend}", url=existing_url, created=False)

    if backend == "llama_cpp":
        raise ValueError("llama_cpp requires LLAMA_CPP_ENDPOINT_URL in current implementation.")

    if not config.create_endpoints:
        raise ValueError(f"{backend} endpoint URL missing and CREATE_ENDPOINTS=false")

    name = _endpoint_name(config, backend, run_id)
    append_jsonl(logs_path, {"event": "endpoint_create_start", "backend": backend, "name": name})

    repository = config.model_id
    try:
        endpoint = api.create_inference_endpoint(
            name=name,
            namespace=config.hf_namespace or None,
            repository=repository,
            framework="pytorch",
            task="text-generation",
            accelerator="gpu",
            vendor="aws",
            region="us-east-1",
            type="protected",
            instance_size="x1",
            instance_type="nvidia-a10g",
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "restricted" in message.lower() and config.fallback_model_id and config.fallback_model_id != repository:
            append_jsonl(
                logs_path,
                {
                    "event": "endpoint_create_retry_with_fallback_model",
                    "backend": backend,
                    "original_model": repository,
                    "fallback_model": config.fallback_model_id,
                },
            )
            repository = config.fallback_model_id
            endpoint = api.create_inference_endpoint(
                name=name,
                namespace=config.hf_namespace or None,
                repository=repository,
                framework="pytorch",
                task="text-generation",
                accelerator="gpu",
                vendor="aws",
                region="us-east-1",
                type="protected",
                instance_size="x1",
                instance_type="nvidia-a10g",
            )
        else:
            raise
    endpoint.wait(timeout=1200)
    append_jsonl(
        logs_path,
        {"event": "endpoint_create_ready", "backend": backend, "name": name, "repository": repository},
    )
    return EndpointInfo(name=name, url=endpoint.url, created=True)


def resolve_endpoints(config: AppConfig, run_id: str, run_dir: Path) -> dict[str, EndpointInfo]:
    logs_path = run_dir / "logs" / "endpoint_manager.jsonl"
    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=config.hf_token) if config.hf_token else HfApi()
    endpoints: dict[str, EndpointInfo] = {}
    for backend in config.engines:
        endpoints[backend] = create_endpoint_if_needed(api, config, backend, run_id, logs_path)
    return endpoints


def apply_shutdown_mode(config: AppConfig, endpoints: dict[str, Any], run_dir: Path) -> None:
    if config.shutdown_mode == "none":
        return

    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=config.hf_token) if config.hf_token else HfApi()
    logs_path = run_dir / "logs" / "endpoint_shutdown.jsonl"

    for backend, info in endpoints.items():
        if not info:
            continue
        name = info.get("name") if isinstance(info, dict) else getattr(info, "name", "")
        created = info.get("created") if isinstance(info, dict) else getattr(info, "created", False)
        if not created or not name or name.startswith("manual-"):
            append_jsonl(logs_path, {"event": "skip_shutdown_manual", "backend": backend, "name": name})
            continue
        endpoint = api.get_inference_endpoint(name=name, namespace=config.hf_namespace or None)
        if config.shutdown_mode == "pause":
            endpoint.pause()
        elif config.shutdown_mode == "scale_to_zero":
            endpoint.scale_to_zero()
        elif config.shutdown_mode == "delete":
            endpoint.delete()
        append_jsonl(
            logs_path,
            {"event": "shutdown_applied", "backend": backend, "name": name, "mode": config.shutdown_mode},
        )
