#!/usr/bin/env bash
# Single public benchmark: GSM8K only, full LIMIT on that task (no per-task split).
# numeric grading is end-to-end; disable custom + lm_eval so one comparable stream.
#
# Defaults: llama.cpp baseline + vLLM/SGLang optimization modes; conditions run in
# parallel (see RUN_BACKENDS_SEQUENTIALLY in src/main.py). Override ENGINES or set
# RUN_BACKENDS_SEQUENTIALLY=true if your endpoints rate-limit under concurrency.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate 2>/dev/null || true

export PUBLIC_TASK_IDS=gsm8k_main
export CUSTOM_WORKLOAD_ENABLED=false
export STANDARD_EVAL_ENABLED=false
export PUBLIC_BENCHMARK_ENABLED=true
export LIMIT="${LIMIT:-300}"
export OPTIMIZATION_MODES="${OPTIMIZATION_MODES:-baseline,kv_cache_quant,spec_decode}"
export ENGINES="${ENGINES:-llama_cpp,vllm,sglang}"
export RUN_BACKENDS_SEQUENTIALLY="${RUN_BACKENDS_SEQUENTIALLY:-false}"

python -m src.main --mode all
