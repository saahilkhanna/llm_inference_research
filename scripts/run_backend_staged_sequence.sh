#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

RESULTS_DIR="${RESULTS_DIR:-real_results/USED_RESULTS}"
FINAL_SUMMARY_DIR="${FINAL_SUMMARY_DIR:-final_summary}"
DATA_COLLECTION_DIR="${DATA_COLLECTION_DIR:-final_summary/data_collection}"
RUN_PREFIX="${RUN_PREFIX:-final_summary}"
LIMIT="${LIMIT:-80}"
STANDARD_EVAL_LIMIT_OVERRIDE="${STANDARD_EVAL_LIMIT_OVERRIDE:-20}"

mkdir -p "$RESULTS_DIR" "$FINAL_SUMMARY_DIR"

latest_run_dir() {
  local run_name="$1"
  python - <<'PY' "$RESULTS_DIR" "$run_name"
from pathlib import Path
import sys
base=Path(sys.argv[1]); run_name=sys.argv[2]
matches=sorted(base.glob(f"*_{run_name}"))
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

run_backend_stage() {
  local stage="$1"
  local engines="$2"
  local modes="$3"
  local run_name="$4"
  local backend_for_export="$5"

  echo "[$stage] start"
  env \
    RUN_NAME="$run_name" \
    ENGINES="$engines" \
    OPTIMIZATION_MODES="$modes" \
    CREATE_ENDPOINTS="true" \
    SHUTDOWN_MODE="pause" \
    PARALLEL_ENDPOINT_PROVISION="true" \
    RUN_BACKENDS_SEQUENTIALLY="false" \
    PUBLIC_BENCHMARK_ENABLED="true" \
    PUBLIC_TASK_IDS="${PUBLIC_TASK_IDS:-gsm8k_main,competition_math,humaneval_coding,longbench_v2_slice}" \
    CUSTOM_WORKLOAD_ENABLED="false" \
    STANDARD_EVAL_ENABLED="true" \
    STANDARD_EVAL_TASKS="${STANDARD_EVAL_TASKS:-gsm8k,humaneval_instruct}" \
    STANDARD_EVAL_RUN_ON_ALL_CONDITIONS="true" \
    LIMIT="$LIMIT" \
    STANDARD_EVAL_LIMIT_OVERRIDE="$STANDARD_EVAL_LIMIT_OVERRIDE" \
    RESULTS_DIR="$RESULTS_DIR" \
    python -m src.main --mode all

  local rdir
  rdir="$(latest_run_dir "$run_name")"
  python scripts/verify_run_integrity.py --run-dir "$rdir"
  python scripts/organize_final_summary.py --run-dir "$rdir" --backend "$backend_for_export" --output-dir "$FINAL_SUMMARY_DIR"
  python scripts/build_data_collection.py --results-dir "$RESULTS_DIR" --output-dir "$DATA_COLLECTION_DIR" --run-dir "$rdir"
  echo "[$stage] completed: $rdir"
}

# 1) Control: llama.cpp baseline only (1 endpoint)
run_backend_stage \
  "llama_cpp_only" \
  "llama_cpp" \
  "baseline" \
  "${RUN_PREFIX}_llama_cpp" \
  "llama_cpp"

# 2) vLLM all modes (max 3 endpoints)
run_backend_stage \
  "vllm_all_modes" \
  "vllm" \
  "baseline,kv_cache_quant,spec_decode" \
  "${RUN_PREFIX}_vllm" \
  "vllm"

# 3) SGLang all modes (max 3 endpoints)
run_backend_stage \
  "sglang_all_modes" \
  "sglang" \
  "baseline,kv_cache_quant,spec_decode" \
  "${RUN_PREFIX}_sglang" \
  "sglang"

echo "[sequence] complete"
echo "[sequence] final summary root: ${FINAL_SUMMARY_DIR}"
echo "[sequence] aggregate: ${FINAL_SUMMARY_DIR}/aggregated_results/aggregated_results.csv"
echo "[sequence] data collection: ${DATA_COLLECTION_DIR}/aggregated"
