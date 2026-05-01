from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Handle types
def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _to_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _to_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def _to_args(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return default
    return shlex.split(value)


def _env_or(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _vllm_spec_decode_args(base_args: list[str], draft_model: str, num_speculative_tokens: int) -> list[str]:
    # vLLM 0.16+ parses this JSON into SpeculativeConfig; `method` must be set (else Pydantic leaves it None → crash).
    cfg: dict[str, Any] = {
        "method": "draft_model",
        "num_speculative_tokens": num_speculative_tokens,
    }
    if draft_model:
        cfg["model"] = draft_model
        
    # vLLM>=0.16 expects speculative decoding knobs via `--speculative_config JSON` (underscore flag).
    speculative = json.dumps(cfg, separators=(",", ":"), ensure_ascii=True)
    return [*base_args, "--speculative_config", speculative]


def _sglang_spec_decode_args(base_args: list[str], draft_model: str, num_steps: int) -> list[str]:
    args = [*base_args]
    
    # SGLang v0.5.x uses `--speculative-draft-model-path` (alias `--speculative-draft-model` also exists).
    if draft_model:
        args.extend(
            [
                "--speculative-algorithm",
                "STANDALONE",
                "--speculative-draft-model-path",
                draft_model,
                # Must be set explicitly: auto-tuned None breaks __post_init__ (eagle_topk vs int).
                "--speculative-eagle-topk",
                "1",
            ]
        )
    args.extend(["--speculative-num-steps", str(num_steps)])
    return args


def _sglang_kv_cache_dtype_for_launcher() -> str:
    explicit = os.getenv("SGLANG_KV_CACHE_QUANT_DTYPE", "").strip()
    if explicit:
        return explicit
    mode = os.getenv("KV_CACHE_QUANT_MODE", "fp8").strip()
    # SGLang v0.5.x rejects bare `fp8`; choices include fp8_e4m3, fp8_e5m2, bf16, ...
    if mode == "fp8":
        return "fp8_e4m3"
    return mode


@dataclass
class AppConfig:
    hf_token: str
    hf_namespace: str
    model_id: str
    llama_cpp_model_id: str
    vllm_model_id: str
    sglang_model_id: str
    fallback_model_id: str
    vllm_endpoint_url: str
    sglang_endpoint_url: str
    vllm_endpoint_url_baseline: str
    vllm_endpoint_url_kv_cache_quant: str
    vllm_endpoint_url_spec_decode: str
    sglang_endpoint_url_baseline: str
    sglang_endpoint_url_kv_cache_quant: str
    sglang_endpoint_url_spec_decode: str
    llama_cpp_endpoint_framework: str
    vllm_endpoint_framework: str
    sglang_endpoint_framework: str
    llama_cpp_engine_image_url: str
    llama_cpp_engine_port: int
    llama_cpp_engine_health_route: str
    llama_cpp_gguf_filename: str
    llama_cpp_ctx_size: int
    llama_cpp_n_parallel: int
    llama_cpp_threads_http: int
    llama_cpp_n_gpu_layers: int | None
    llama_cpp_gguf_variant: str | None
    endpoint_accelerator: str
    endpoint_vendor: str
    endpoint_region: str
    endpoint_type: str
    endpoint_instance_size: str
    endpoint_instance_type: str
    endpoint_min_replica: int
    endpoint_max_replica: int
    endpoint_scale_to_zero_timeout: int
    endpoint_hardware_usage_target: int
    endpoint_cache_http_responses: bool
    endpoint_notifications_email: bool
    endpoint_notifications_push: bool
    endpoint_tensor_parallel_size: int
    endpoint_ready_timeout_seconds: int
    vllm_engine_image_url: str
    sglang_engine_image_url: str
    vllm_engine_port: int
    sglang_engine_port: int
    vllm_engine_health_route: str
    sglang_engine_health_route: str
    llama_cpp_args_baseline: list[str]
    vllm_args_baseline: list[str]
    vllm_args_kv_cache_quant: list[str]
    vllm_args_spec_decode: list[str]
    sglang_args_baseline: list[str]
    sglang_args_kv_cache_quant: list[str]
    sglang_args_spec_decode: list[str]
    create_endpoints: bool
    shutdown_mode: str
    provision_endpoints_parallel: bool
    run_backends_sequentially: bool
    limit: int
    smoke_limit: int
    temperature: float
    top_p: float
    max_new_tokens: int
    engines: list[str]
    optimization_modes: list[str]
    repeats_per_condition: int
    llama_cpp_endpoint_url: str
    kv_cache_quant_mode: str
    speculative_draft_model: str
    speculative_num_tokens: int
    apply_optimization_request_hints: bool
    standard_eval_enabled: bool
    standard_evaluator: str
    standard_eval_tasks: list[str]
    standard_eval_run_on_all_conditions: bool
    standard_eval_limit_override: int
    standard_eval_humaneval_code_only_prompt: bool
    standard_eval_humaneval_system_prompt: str
    public_task_ids: list[str]
    public_benchmark_enabled: bool
    custom_workload_enabled: bool
    custom_workload_size: int
    results_dir: str
    run_name: str
    max_estimated_cost_usd: float
    aiperf_enabled: bool
    aiperf_synthetic_enabled: bool
    request_timeout_seconds: int
    backend_api_style: str
    public_tasks: list[dict[str, Any]] = field(default_factory=list)
    custom_workload: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

    def effective_limit(self, smoke: bool) -> int:
        return self.smoke_limit if smoke else self.limit

    def model_id_for_backend(self, backend: str) -> str:
        if backend == "llama_cpp":
            return self.llama_cpp_model_id
        if backend == "vllm":
            return self.vllm_model_id
        if backend == "sglang":
            return self.sglang_model_id
        return self.model_id

    def endpoint_args_for(self, backend: str, optimization_mode: str) -> list[str]:
        if backend == "llama_cpp":
            return self.llama_cpp_args_baseline
        if backend == "vllm":
            by_mode = {
                "baseline": self.vllm_args_baseline,
                "kv_cache_quant": self.vllm_args_kv_cache_quant,
                "spec_decode": self.vllm_args_spec_decode,
            }
            return by_mode.get(optimization_mode, self.vllm_args_baseline)
        if backend == "sglang":
            by_mode = {
                "baseline": self.sglang_args_baseline,
                "kv_cache_quant": self.sglang_args_kv_cache_quant,
                "spec_decode": self.sglang_args_spec_decode,
            }
            return by_mode.get(optimization_mode, self.sglang_args_baseline)
        return []

    def endpoint_url_for_mode(self, backend: str, optimization_mode: str) -> str:
        if backend == "vllm":
            by_mode = {
                "baseline": self.vllm_endpoint_url_baseline,
                "kv_cache_quant": self.vllm_endpoint_url_kv_cache_quant,
                "spec_decode": self.vllm_endpoint_url_spec_decode,
            }
            return by_mode.get(optimization_mode, "") or self.vllm_endpoint_url
        if backend == "sglang":
            by_mode = {
                "baseline": self.sglang_endpoint_url_baseline,
                "kv_cache_quant": self.sglang_endpoint_url_kv_cache_quant,
                "spec_decode": self.sglang_endpoint_url_spec_decode,
            }
            return by_mode.get(optimization_mode, "") or self.sglang_endpoint_url
        if backend == "llama_cpp":
            return self.llama_cpp_endpoint_url if optimization_mode == "baseline" else ""
        return ""

    def validate(self) -> None:
        allowed_engines = {"llama_cpp", "vllm", "sglang"}
        if not self.engines:
            raise ValueError("ENGINES must include at least one engine.")
        unknown_engines = [e for e in self.engines if e not in allowed_engines]
        if unknown_engines:
            raise ValueError(f"Unsupported engines: {unknown_engines}. Allowed: {sorted(allowed_engines)}")

        allowed_modes = {"baseline", "kv_cache_quant", "spec_decode"}
        if not self.optimization_modes:
            raise ValueError("OPTIMIZATION_MODES must include at least one mode.")
        unknown_modes = [m for m in self.optimization_modes if m not in allowed_modes]
        if unknown_modes:
            raise ValueError(f"Unsupported OPTIMIZATION_MODES: {unknown_modes}. Allowed: {sorted(allowed_modes)}")

        if self.repeats_per_condition <= 0:
            raise ValueError("REPEATS_PER_CONDITION must be > 0.")
        if self.standard_evaluator not in {"lm_eval", "none"}:
            raise ValueError("STANDARD_EVALUATOR must be one of lm_eval|none.")
        if self.standard_eval_limit_override < 0:
            raise ValueError("STANDARD_EVAL_LIMIT_OVERRIDE must be >= 0.")

        if self.create_endpoints and not self.hf_token:
            raise ValueError("HF_TOKEN is required when CREATE_ENDPOINTS=true.")
        if not self.create_endpoints:
            has_any_vllm_url = any(
                [
                    self.vllm_endpoint_url,
                    self.vllm_endpoint_url_baseline,
                    self.vllm_endpoint_url_kv_cache_quant,
                    self.vllm_endpoint_url_spec_decode,
                ]
            )
            has_any_sglang_url = any(
                [
                    self.sglang_endpoint_url,
                    self.sglang_endpoint_url_baseline,
                    self.sglang_endpoint_url_kv_cache_quant,
                    self.sglang_endpoint_url_spec_decode,
                ]
            )
            if "vllm" in self.engines and not has_any_vllm_url:
                raise ValueError("Set VLLM_ENDPOINT_URL when running vllm in manual mode.")
            if "sglang" in self.engines and not has_any_sglang_url:
                raise ValueError("Set SGLANG_ENDPOINT_URL when running sglang in manual mode.")
            
            # Integrity guard: with optimization experiments in manual mode, require explicit
            # per-mode endpoint URLs so each condition maps to an intentionally configured server.
            for backend in ("vllm", "sglang"):
                if backend not in self.engines:
                    continue
                active_modes = [m for m in self.optimization_modes if backend != "llama_cpp" or m == "baseline"]
                if len(active_modes) <= 1:
                    continue
                resolved_urls = []
                for mode in active_modes:
                    url = self.endpoint_url_for_mode(backend, mode).strip()
                    if not url:
                        raise ValueError(
                            f"Manual mode requires explicit {backend} endpoint URL for optimization mode '{mode}'. "
                            f"Set {backend.upper()}_ENDPOINT_URL_{mode.upper()}."
                        )
                    resolved_urls.append((mode, url))
                unique_urls = {u for _, u in resolved_urls}
                if len(unique_urls) != len(resolved_urls):
                    pairs = ", ".join([f"{m}={u}" for m, u in resolved_urls])
                    raise ValueError(
                        f"Manual mode requires distinct {backend} endpoint URLs per optimization mode. "
                        f"Resolved mappings: {pairs}"
                    )
        if (
            "llama_cpp" in self.engines
            and not self.llama_cpp_endpoint_url
            and not self.create_endpoints
        ):
            raise ValueError("Set LLAMA_CPP_ENDPOINT_URL or CREATE_ENDPOINTS=true when ENGINES includes llama_cpp.")
        if self.shutdown_mode not in {"pause", "delete", "scale_to_zero", "none"}:
            raise ValueError("SHUTDOWN_MODE must be one of pause|delete|scale_to_zero|none.")
        if self.backend_api_style not in {"auto", "openai_chat", "tgi"}:
            raise ValueError("BACKEND_API_STYLE must be one of auto|openai_chat|tgi.")
        if self.endpoint_type not in {"protected", "public", "private"}:
            raise ValueError("ENDPOINT_TYPE must be one of protected|public|private.")
        if self.max_new_tokens <= 0:
            raise ValueError("MAX_NEW_TOKENS must be > 0.")
        if self.effective_limit(False) <= 0 or self.smoke_limit <= 0:
            raise ValueError("LIMIT and SMOKE_LIMIT must be > 0.")
        if self.endpoint_ready_timeout_seconds < 60:
            raise ValueError("ENDPOINT_READY_TIMEOUT_SECONDS must be >= 60.")

def load_config(env_path: str = ".env", defaults_path: str = "config/default.yaml") -> AppConfig:
    load_dotenv(env_path, override=False)

    defaults: dict[str, Any] = {}
    defaults_file = Path(defaults_path)
    if defaults_file.exists():
        with defaults_file.open("r", encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}

    target_llm_base_url = os.getenv("TARGET_LLM_BASE_URL", "").strip()
    vllm_url = os.getenv("VLLM_ENDPOINT_URL", "").strip()
    sglang_url = os.getenv("SGLANG_ENDPOINT_URL", "").strip()
    create_endpoints_value = os.getenv("CREATE_ENDPOINTS")
    create_endpoints_default = _to_bool(create_endpoints_value, True)

    # This is our convenience alias: allow a single endpoint URL var for quick manual vLLM/SGLang runs.
    # llama.cpp remains explicit because it is normally a separate local server baseline.
    if target_llm_base_url and not vllm_url and not sglang_url:
        vllm_url = target_llm_base_url
        sglang_url = target_llm_base_url
        if create_endpoints_value is None:
            create_endpoints_default = False

    default_model = _env_or("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    llama_cpp_model_id = os.getenv("LLAMA_CPP_MODEL_ID", "").strip() or default_model
    vllm_model_id = os.getenv("VLLM_MODEL_ID", "").strip() or default_model
    sglang_model_id = os.getenv("SGLANG_MODEL_ID", "").strip() or default_model

    cfg = AppConfig(
        hf_token=os.getenv("HF_TOKEN", ""),
        hf_namespace=os.getenv("HF_NAMESPACE", ""),
        model_id=default_model,
        llama_cpp_model_id=llama_cpp_model_id,
        vllm_model_id=vllm_model_id,
        sglang_model_id=sglang_model_id,
        fallback_model_id=os.getenv("FALLBACK_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        vllm_endpoint_url=vllm_url,
        sglang_endpoint_url=sglang_url,
        vllm_endpoint_url_baseline=os.getenv("VLLM_ENDPOINT_URL_BASELINE", "").strip(),
        vllm_endpoint_url_kv_cache_quant=os.getenv("VLLM_ENDPOINT_URL_KV_CACHE_QUANT", "").strip(),
        vllm_endpoint_url_spec_decode=os.getenv("VLLM_ENDPOINT_URL_SPEC_DECODE", "").strip(),
        sglang_endpoint_url_baseline=os.getenv("SGLANG_ENDPOINT_URL_BASELINE", "").strip(),
        sglang_endpoint_url_kv_cache_quant=os.getenv("SGLANG_ENDPOINT_URL_KV_CACHE_QUANT", "").strip(),
        sglang_endpoint_url_spec_decode=os.getenv("SGLANG_ENDPOINT_URL_SPEC_DECODE", "").strip(),
        llama_cpp_endpoint_framework=os.getenv("LLAMA_CPP_ENDPOINT_FRAMEWORK", "llamacpp").strip(),
        vllm_endpoint_framework=os.getenv("VLLM_ENDPOINT_FRAMEWORK", "pytorch").strip(),
        sglang_endpoint_framework=os.getenv("SGLANG_ENDPOINT_FRAMEWORK", "pytorch").strip(),
        llama_cpp_engine_image_url=os.getenv(
            "LLAMA_CPP_ENGINE_IMAGE_URL", "ghcr.io/ggml-org/llama.cpp:server-cuda"
        ).strip(),
        llama_cpp_engine_port=_to_int(os.getenv("LLAMA_CPP_ENGINE_PORT"), 8080),
        llama_cpp_engine_health_route=os.getenv("LLAMA_CPP_ENGINE_HEALTH_ROUTE", "/health"),
        llama_cpp_gguf_filename=os.getenv("LLAMA_CPP_GGUF_FILENAME", "").strip(),
        llama_cpp_ctx_size=_to_int(os.getenv("LLAMA_CPP_CTX_SIZE"), 8096),
        llama_cpp_n_parallel=_to_int(os.getenv("LLAMA_CPP_N_PARALLEL"), 1),
        llama_cpp_threads_http=_to_int(os.getenv("LLAMA_CPP_THREADS_HTTP"), 64),
        llama_cpp_n_gpu_layers=(
            None
            if os.getenv("LLAMA_CPP_N_GPU_LAYERS") is None
            else (
                None
                if os.getenv("LLAMA_CPP_N_GPU_LAYERS", "").strip().lower()
                in {"", "none", "null"}
                else _to_int(os.getenv("LLAMA_CPP_N_GPU_LAYERS"), 9999)
            )
        ),
        llama_cpp_gguf_variant=os.getenv("LLAMA_CPP_GGUF_VARIANT", "").strip() or None,
        endpoint_accelerator=os.getenv("ENDPOINT_ACCELERATOR", "gpu"),
        endpoint_vendor=os.getenv("ENDPOINT_VENDOR", "aws"),
        endpoint_region=os.getenv("ENDPOINT_REGION", "us-east-1"),
        endpoint_type=os.getenv("ENDPOINT_TYPE", "protected"),
        endpoint_instance_size=os.getenv("ENDPOINT_INSTANCE_SIZE", "x1"),
        endpoint_instance_type=os.getenv("ENDPOINT_INSTANCE_TYPE", "nvidia-l4"),
        endpoint_min_replica=_to_int(os.getenv("ENDPOINT_MIN_REPLICA"), 0),
        endpoint_max_replica=_to_int(os.getenv("ENDPOINT_MAX_REPLICA"), 1),
        endpoint_scale_to_zero_timeout=_to_int(os.getenv("ENDPOINT_SCALE_TO_ZERO_TIMEOUT"), 60),
        endpoint_hardware_usage_target=_to_int(os.getenv("ENDPOINT_HARDWARE_USAGE_TARGET"), 80),
        endpoint_cache_http_responses=_to_bool(os.getenv("ENDPOINT_CACHE_HTTP_RESPONSES"), False),
        endpoint_notifications_email=_to_bool(os.getenv("ENDPOINT_NOTIFICATIONS_EMAIL"), True),
        endpoint_notifications_push=_to_bool(os.getenv("ENDPOINT_NOTIFICATIONS_PUSH"), True),
        endpoint_tensor_parallel_size=_to_int(os.getenv("ENDPOINT_TENSOR_PARALLEL_SIZE"), 1),
        endpoint_ready_timeout_seconds=_to_int(os.getenv("ENDPOINT_READY_TIMEOUT_SECONDS"), 1800),
        vllm_engine_image_url=os.getenv("VLLM_ENGINE_IMAGE_URL", "vllm/vllm-openai:v0.16.0"),
        sglang_engine_image_url=os.getenv("SGLANG_ENGINE_IMAGE_URL", "lmsysorg/sglang:v0.5.8"),
        vllm_engine_port=_to_int(os.getenv("VLLM_ENGINE_PORT"), 8000),
        sglang_engine_port=_to_int(os.getenv("SGLANG_ENGINE_PORT"), 30000),
        vllm_engine_health_route=os.getenv("VLLM_ENGINE_HEALTH_ROUTE", "/health"),
        sglang_engine_health_route=os.getenv("SGLANG_ENGINE_HEALTH_ROUTE", "/health"),
        llama_cpp_args_baseline=_to_args(os.getenv("LLAMA_CPP_BASELINE_ARGS"), []),
        vllm_args_baseline=_to_args(
            os.getenv("VLLM_BASELINE_ARGS"),
            ["--gpu-memory-utilization", "0.90", "--max-model-len", "8096"],
        ),
        vllm_args_kv_cache_quant=_to_args(
            os.getenv("VLLM_KV_CACHE_QUANT_ARGS"),
            [
                "--gpu-memory-utilization",
                "0.90",
                "--max-model-len",
                "8096",
                "--kv-cache-dtype",
                os.getenv("KV_CACHE_QUANT_MODE", "fp8"),
            ],
        ),
        vllm_args_spec_decode=_to_args(
            os.getenv("VLLM_SPEC_DECODE_ARGS"),
            _vllm_spec_decode_args(
                ["--gpu-memory-utilization", "0.90", "--max-model-len", "8096"],
                os.getenv("SPECULATIVE_DRAFT_MODEL", "").strip()
                or _env_or(
                    "SPECULATIVE_DRAFT_MODEL_DEFAULT",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ),
                _to_int(os.getenv("SPECULATIVE_NUM_TOKENS"), 5),
            ),
        ),
        sglang_args_baseline=_to_args(
            os.getenv("SGLANG_BASELINE_ARGS"),
            ["--trust-remote-code", "--mem-fraction-static", "0.90", "--max-total-tokens", "8096"],
        ),
        sglang_args_kv_cache_quant=_to_args(
            os.getenv("SGLANG_KV_CACHE_QUANT_ARGS"),
            [
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.90",
                "--max-total-tokens",
                "8096",
                "--kv-cache-dtype",
                _sglang_kv_cache_dtype_for_launcher(),
            ],
        ),
        sglang_args_spec_decode=_to_args(
            os.getenv("SGLANG_SPEC_DECODE_ARGS"),
            _sglang_spec_decode_args(
                ["--trust-remote-code", "--mem-fraction-static", "0.90", "--max-total-tokens", "8096"],
                os.getenv("SPECULATIVE_DRAFT_MODEL", "").strip()
                or _env_or(
                    "SPECULATIVE_DRAFT_MODEL_DEFAULT",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ),
                _to_int(os.getenv("SPECULATIVE_NUM_TOKENS"), 5),
            ),
        ),
        create_endpoints=create_endpoints_default,
        shutdown_mode=os.getenv("SHUTDOWN_MODE", "pause"),
        provision_endpoints_parallel=_to_bool(os.getenv("PARALLEL_ENDPOINT_PROVISION"), True),
        run_backends_sequentially=_to_bool(os.getenv("RUN_BACKENDS_SEQUENTIALLY"), True),
        limit=_to_int(os.getenv("LIMIT"), 100),
        smoke_limit=_to_int(os.getenv("SMOKE_LIMIT"), 8),
        temperature=_to_float(os.getenv("TEMPERATURE"), 0.0),
        top_p=_to_float(os.getenv("TOP_P"), 1.0),
        max_new_tokens=_to_int(os.getenv("MAX_NEW_TOKENS"), 256),
        engines=_to_list(os.getenv("ENGINES"), ["vllm", "sglang"]),
        optimization_modes=_to_list(os.getenv("OPTIMIZATION_MODES"), ["baseline"]),
        repeats_per_condition=_to_int(os.getenv("REPEATS_PER_CONDITION"), 1),
        llama_cpp_endpoint_url=os.getenv("LLAMA_CPP_ENDPOINT_URL", "").strip(),
        kv_cache_quant_mode=os.getenv("KV_CACHE_QUANT_MODE", "fp8"),
        speculative_draft_model=os.getenv("SPECULATIVE_DRAFT_MODEL", ""),
        speculative_num_tokens=_to_int(os.getenv("SPECULATIVE_NUM_TOKENS"), 5),
        apply_optimization_request_hints=_to_bool(os.getenv("APPLY_OPTIMIZATION_REQUEST_HINTS"), False),
        standard_eval_enabled=_to_bool(os.getenv("STANDARD_EVAL_ENABLED"), True),
        standard_evaluator=os.getenv("STANDARD_EVALUATOR", "lm_eval"),
        standard_eval_tasks=_to_list(os.getenv("STANDARD_EVAL_TASKS"), ["gsm8k"]),
        standard_eval_run_on_all_conditions=_to_bool(os.getenv("STANDARD_EVAL_RUN_ON_ALL_CONDITIONS"), False),
        standard_eval_limit_override=_to_int(os.getenv("STANDARD_EVAL_LIMIT_OVERRIDE"), 0),
        standard_eval_humaneval_code_only_prompt=_to_bool(
            os.getenv("STANDARD_EVAL_HUMANEVAL_CODE_ONLY_PROMPT"), True
        ),
        standard_eval_humaneval_system_prompt=os.getenv(
            "STANDARD_EVAL_HUMANEVAL_SYSTEM_PROMPT",
            "Return only valid Python code for the target function. No markdown fences. No prose.",
        ).strip(),
        public_task_ids=_to_list(os.getenv("PUBLIC_TASK_IDS"), []),
        public_benchmark_enabled=_to_bool(os.getenv("PUBLIC_BENCHMARK_ENABLED"), True),
        custom_workload_enabled=_to_bool(os.getenv("CUSTOM_WORKLOAD_ENABLED"), True),
        custom_workload_size=_to_int(os.getenv("CUSTOM_WORKLOAD_SIZE"), 45),
        results_dir=os.getenv("RESULTS_DIR", "results"),
        run_name=os.getenv("RUN_NAME", "black_box_inference_backends"),
        max_estimated_cost_usd=_to_float(os.getenv("MAX_ESTIMATED_COST_USD"), 10.0),
        aiperf_enabled=_to_bool(os.getenv("AIPERF_ENABLED"), True),
        aiperf_synthetic_enabled=_to_bool(os.getenv("AIPERF_SYNTHETIC_ENABLED"), True),
        request_timeout_seconds=_to_int(os.getenv("REQUEST_TIMEOUT_SECONDS"), 120),
        backend_api_style=os.getenv("BACKEND_API_STYLE", "auto"),
        public_tasks=defaults.get("public_tasks", []),
        custom_workload=defaults.get("custom_workload", {}),
        generation=defaults.get("generation", {}),
        analysis=defaults.get("analysis", {}),
    )
    cfg.validate()
    return cfg
