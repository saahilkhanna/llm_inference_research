#!/usr/bin/env bash
# Single public workload: Hugging Face `THUDM/LongBench-v2` (YAML id longbench_v2_slice).
# Rows are filtered by LONGBENCH_MAX_CONTEXT_CHARS then shuffled with LONGBENCH_SELECTION_SEED.
# MCQ grading (A–D); contexts are tens of thousands of chars — raise LONG_CONTEXT_MAX_SEQ_LEN or
# lower LONGBENCH_MAX_CONTEXT_CHARS so inputs fit endpoint max-model-len without truncation.
#
# Defaults match the GSM8K grid: llama.cpp + vLLM + SGLang, three optimization modes,
# parallel conditions unless RUN_BACKENDS_SEQUENTIALLY=true.
# To keep pinned VLLM/SGLang/LLAMA_CPP args from `.env`, set LONGBENCH_KEEP_ENGINE_ARGS=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source .venv/bin/activate 2>/dev/null || true

export PUBLIC_TASK_IDS=longbench_v2_slice
export CUSTOM_WORKLOAD_ENABLED=false
export STANDARD_EVAL_ENABLED=false
export PUBLIC_BENCHMARK_ENABLED=true
export LIMIT="${LIMIT:-150}"

# Real LongBench-v2 prompts are ~50–120k UTF-8 chars; endpoint max-model-len must match or inputs truncate.
# If .env pins VLLM_*_ARGS / SGLANG_*_ARGS with small --max-model-len, defaults here never apply (python-dotenv
# merge). Unless LONGBENCH_KEEP_ENGINE_ARGS=1, drop pinned engine arg lists so LONG_CONTEXT_* governs configs.
if [[ "${LONGBENCH_KEEP_ENGINE_ARGS:-0}" != "1" ]]; then
	unset VLLM_BASELINE_ARGS VLLM_KV_CACHE_QUANT_ARGS VLLM_SPEC_DECODE_ARGS \
		SGLANG_BASELINE_ARGS SGLANG_KV_CACHE_QUANT_ARGS SGLANG_SPEC_DECODE_ARGS LLAMA_CPP_BASELINE_ARGS \
		LLAMA_CPP_CTX_SIZE 2>/dev/null || true
fi

export LONGBENCH_MAX_CONTEXT_CHARS="${LONGBENCH_MAX_CONTEXT_CHARS:-120000}"
export LONG_CONTEXT_MAX_SEQ_LEN="${LONG_CONTEXT_MAX_SEQ_LEN:-32768}"
export REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"

export OPTIMIZATION_MODES="${OPTIMIZATION_MODES:-baseline,kv_cache_quant,spec_decode}"
export ENGINES="${ENGINES:-llama_cpp,vllm,sglang}"
export RUN_BACKENDS_SEQUENTIALLY="${RUN_BACKENDS_SEQUENTIALLY:-false}"

python -m src.main --mode all
