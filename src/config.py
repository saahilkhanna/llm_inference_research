from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


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


@dataclass
class AppConfig:
    hf_token: str
    hf_namespace: str
    model_id: str
    fallback_model_id: str
    vllm_endpoint_url: str
    sglang_endpoint_url: str
    vllm_endpoint_url_baseline: str
    vllm_endpoint_url_kv_cache_quant: str
    vllm_endpoint_url_spec_decode: str
    sglang_endpoint_url_baseline: str
    sglang_endpoint_url_kv_cache_quant: str
    sglang_endpoint_url_spec_decode: str
    create_endpoints: bool
    shutdown_mode: str
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
        if "llama_cpp" in self.engines and not self.llama_cpp_endpoint_url:
            raise ValueError("Set LLAMA_CPP_ENDPOINT_URL when ENGINES includes llama_cpp.")
        if self.shutdown_mode not in {"pause", "delete", "scale_to_zero", "none"}:
            raise ValueError("SHUTDOWN_MODE must be one of pause|delete|scale_to_zero|none.")
        if self.backend_api_style not in {"auto", "openai_chat", "tgi"}:
            raise ValueError("BACKEND_API_STYLE must be one of auto|openai_chat|tgi.")
        if self.max_new_tokens <= 0:
            raise ValueError("MAX_NEW_TOKENS must be > 0.")
        if self.effective_limit(False) <= 0 or self.smoke_limit <= 0:
            raise ValueError("LIMIT and SMOKE_LIMIT must be > 0.")

    def endpoint_url_for(self, engine: str, optimization_mode: str, default_url: str) -> str:
        if engine == "vllm":
            by_mode = {
                "baseline": self.vllm_endpoint_url_baseline,
                "kv_cache_quant": self.vllm_endpoint_url_kv_cache_quant,
                "spec_decode": self.vllm_endpoint_url_spec_decode,
            }
            return by_mode.get(optimization_mode, "") or self.vllm_endpoint_url or default_url
        if engine == "sglang":
            by_mode = {
                "baseline": self.sglang_endpoint_url_baseline,
                "kv_cache_quant": self.sglang_endpoint_url_kv_cache_quant,
                "spec_decode": self.sglang_endpoint_url_spec_decode,
            }
            return by_mode.get(optimization_mode, "") or self.sglang_endpoint_url or default_url
        return default_url


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
    create_endpoints_default = _to_bool(os.getenv("CREATE_ENDPOINTS"), True)

    # Convenience alias: allow a single endpoint URL var for quick manual runs.
    # If backend-specific URLs are not set, use TARGET_LLM_BASE_URL for both.
    if target_llm_base_url and not vllm_url and not sglang_url:
        vllm_url = target_llm_base_url
        sglang_url = target_llm_base_url
        create_endpoints_default = False

    cfg = AppConfig(
        hf_token=os.getenv("HF_TOKEN", ""),
        hf_namespace=os.getenv("HF_NAMESPACE", ""),
        model_id=os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        fallback_model_id=os.getenv("FALLBACK_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        vllm_endpoint_url=vllm_url,
        sglang_endpoint_url=sglang_url,
        vllm_endpoint_url_baseline=os.getenv("VLLM_ENDPOINT_URL_BASELINE", "").strip(),
        vllm_endpoint_url_kv_cache_quant=os.getenv("VLLM_ENDPOINT_URL_KV_CACHE_QUANT", "").strip(),
        vllm_endpoint_url_spec_decode=os.getenv("VLLM_ENDPOINT_URL_SPEC_DECODE", "").strip(),
        sglang_endpoint_url_baseline=os.getenv("SGLANG_ENDPOINT_URL_BASELINE", "").strip(),
        sglang_endpoint_url_kv_cache_quant=os.getenv("SGLANG_ENDPOINT_URL_KV_CACHE_QUANT", "").strip(),
        sglang_endpoint_url_spec_decode=os.getenv("SGLANG_ENDPOINT_URL_SPEC_DECODE", "").strip(),
        create_endpoints=create_endpoints_default,
        shutdown_mode=os.getenv("SHUTDOWN_MODE", "pause"),
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
        public_benchmark_enabled=_to_bool(os.getenv("PUBLIC_BENCHMARK_ENABLED"), True),
        custom_workload_enabled=_to_bool(os.getenv("CUSTOM_WORKLOAD_ENABLED"), True),
        custom_workload_size=_to_int(os.getenv("CUSTOM_WORKLOAD_SIZE"), 45),
        results_dir=os.getenv("RESULTS_DIR", "results"),
        run_name=os.getenv("RUN_NAME", "black_box_vllm_sglang"),
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
