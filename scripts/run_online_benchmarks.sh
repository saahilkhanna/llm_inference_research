#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

# Online-benchmark-focused profile:
# - disables custom workload
# - emphasizes lm-eval standard tasks
# - keeps optimization comparison enabled by default
export CUSTOM_WORKLOAD_ENABLED="${CUSTOM_WORKLOAD_ENABLED:-false}"
export PUBLIC_BENCHMARK_ENABLED="${PUBLIC_BENCHMARK_ENABLED:-true}"
export AIPERF_ENABLED="${AIPERF_ENABLED:-false}"
export STANDARD_EVAL_ENABLED="${STANDARD_EVAL_ENABLED:-true}"
export STANDARD_EVALUATOR="${STANDARD_EVALUATOR:-lm_eval}"
# Harder online-standard mix for math + coding.
# - gsm8k_platinum: harder GSM8K-style set
# - hendrycks_math500: competition-style math slice
# - minerva_math: stronger math reasoning stress
# - humaneval_instruct: coding functional correctness
export STANDARD_EVAL_TASKS="${STANDARD_EVAL_TASKS:-gsm8k_platinum,hendrycks_math500,minerva_math,humaneval_instruct}"
export STANDARD_EVAL_RUN_ON_ALL_CONDITIONS="${STANDARD_EVAL_RUN_ON_ALL_CONDITIONS:-true}"
export STANDARD_EVAL_LIMIT_OVERRIDE="${STANDARD_EVAL_LIMIT_OVERRIDE:-30}"
export OPTIMIZATION_MODES="${OPTIMIZATION_MODES:-baseline,kv_cache_quant,spec_decode}"
export APPLY_OPTIMIZATION_REQUEST_HINTS="${APPLY_OPTIMIZATION_REQUEST_HINTS:-true}"

python -m src.main --mode all
