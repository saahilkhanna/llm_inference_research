#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

# This is intentionally "all" mode with smoke-sized limits: src.main --mode smoke
# forces OPTIMIZATION_MODES=baseline, which would not create all seven conditions.
export ENGINES="${ENGINES:-llama_cpp,vllm,sglang}"
export OPTIMIZATION_MODES="${OPTIMIZATION_MODES:-baseline,kv_cache_quant,spec_decode}"
export REPEATS_PER_CONDITION="${REPEATS_PER_CONDITION:-1}"
export PUBLIC_BENCHMARK_ENABLED="${PUBLIC_BENCHMARK_ENABLED:-true}"
export PUBLIC_TASK_IDS="${PUBLIC_TASK_IDS:-gsm8k_main}"
export CUSTOM_WORKLOAD_ENABLED="${CUSTOM_WORKLOAD_ENABLED:-false}"
export LIMIT="${LIMIT:-1}"
export SMOKE_LIMIT="${SMOKE_LIMIT:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export STANDARD_EVAL_ENABLED="${STANDARD_EVAL_ENABLED:-false}"
export STANDARD_EVAL_TASKS="${STANDARD_EVAL_TASKS:-gsm8k}"
export STANDARD_EVAL_LIMIT_OVERRIDE="${STANDARD_EVAL_LIMIT_OVERRIDE:-1}"
export STANDARD_EVAL_RUN_ON_ALL_CONDITIONS="${STANDARD_EVAL_RUN_ON_ALL_CONDITIONS:-false}"
export AIPERF_ENABLED="${AIPERF_ENABLED:-false}"
export APPLY_OPTIMIZATION_REQUEST_HINTS="${APPLY_OPTIMIZATION_REQUEST_HINTS:-false}"
export RUN_NAME="${RUN_NAME:-gsm8k_7_endpoint_smoke}"
export RESULTS_DIR="${RESULTS_DIR:-real_results}"
export CREATE_ENDPOINTS="${CREATE_ENDPOINTS:-true}"
export SHUTDOWN_MODE="${SHUTDOWN_MODE:-pause}"
export BACKEND_API_STYLE="${BACKEND_API_STYLE:-openai_chat}"
# Pin GGUF choice for multi-file repos (avoids requiring a successful Hub tree listing at runtime).
export LLAMA_CPP_GGUF_FILENAME="${LLAMA_CPP_GGUF_FILENAME:-Llama-3.2-3B-Instruct-Q4_K_M.gguf}"

python -m src.main --mode all
