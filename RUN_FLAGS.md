# Run Flags and Execution Guide

This is the practical reference for running the current pipeline while keeping integrity checks and run logs.

## Main Commands

```bash
make setup
make smoke
make staged-backends
make single-workload
make online
make all
make shutdown
```

- `staged-backends` runs `llama_cpp` baseline, then `vllm` all modes, then `sglang` all modes.
- `single-workload` keeps only GSM8K (`scripts/run_single_workload_gsm8k.sh`).
- `online` runs public/lm_eval-heavy profile (`scripts/run_online_benchmarks.sh`).

## Core Flags

### Auth and model routing

- `HF_TOKEN`, `HF_NAMESPACE`
- `MODEL_ID`, `FALLBACK_MODEL_ID`
- `LLAMA_CPP_MODEL_ID`, `VLLM_MODEL_ID`, `SGLANG_MODEL_ID`

### Endpoint URLs and provisioning

- `CREATE_ENDPOINTS`
- `LLAMA_CPP_ENDPOINT_URL`
- `VLLM_ENDPOINT_URL`, `SGLANG_ENDPOINT_URL`
- `VLLM_ENDPOINT_URL_BASELINE`, `VLLM_ENDPOINT_URL_KV_CACHE_QUANT`, `VLLM_ENDPOINT_URL_SPEC_DECODE`
- `SGLANG_ENDPOINT_URL_BASELINE`, `SGLANG_ENDPOINT_URL_KV_CACHE_QUANT`, `SGLANG_ENDPOINT_URL_SPEC_DECODE`
- `SHUTDOWN_MODE`
- `PARALLEL_ENDPOINT_PROVISION`
- `RUN_BACKENDS_SEQUENTIALLY`

### Experiment matrix

- `ENGINES` (`llama_cpp,vllm,sglang`)
- `OPTIMIZATION_MODES` (`baseline,kv_cache_quant,spec_decode`)
- `REPEATS_PER_CONDITION`
- `LIMIT`, `SMOKE_LIMIT`
- `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`
- `REQUEST_TIMEOUT_SECONDS`
- `BACKEND_API_STYLE`

### Optimization / launcher args

- `KV_CACHE_QUANT_MODE`
- `SPECULATIVE_DRAFT_MODEL`, `SPECULATIVE_NUM_TOKENS`
- `VLLM_BASELINE_ARGS`, `VLLM_KV_CACHE_QUANT_ARGS`, `VLLM_SPEC_DECODE_ARGS`
- `SGLANG_BASELINE_ARGS`, `SGLANG_KV_CACHE_QUANT_ARGS`, `SGLANG_SPEC_DECODE_ARGS`
- `LLAMA_CPP_BASELINE_ARGS`
- `APPLY_OPTIMIZATION_REQUEST_HINTS`

### Workloads and standard evaluator

- `PUBLIC_BENCHMARK_ENABLED`, `PUBLIC_TASK_IDS`
- `CUSTOM_WORKLOAD_ENABLED`, `CUSTOM_WORKLOAD_SIZE`
- `STANDARD_EVAL_ENABLED`, `STANDARD_EVALUATOR`, `STANDARD_EVAL_TASKS`
- `STANDARD_EVAL_RUN_ON_ALL_CONDITIONS`, `STANDARD_EVAL_LIMIT_OVERRIDE`
- `STANDARD_EVAL_HUMANEVAL_CODE_ONLY_PROMPT`
- `STANDARD_EVAL_HUMANEVAL_SYSTEM_PROMPT`

### Output and profiling

- `RESULTS_DIR`, `RUN_NAME`
- `MAX_ESTIMATED_COST_USD`
- `AIPERF_ENABLED`, `AIPERF_SYNTHETIC_ENABLED`

## Recommended Recipes

### Managed endpoints (7-condition matrix)

```bash
ENGINES=llama_cpp,vllm,sglang \
OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode \
CREATE_ENDPOINTS=true \
make all
```

### Manual endpoints with explicit per-mode URLs

```bash
CREATE_ENDPOINTS=false \
ENGINES=llama_cpp,vllm,sglang \
OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode \
LLAMA_CPP_ENDPOINT_URL=http://localhost:8080 \
VLLM_ENDPOINT_URL_BASELINE=https://... \
VLLM_ENDPOINT_URL_KV_CACHE_QUANT=https://... \
VLLM_ENDPOINT_URL_SPEC_DECODE=https://... \
SGLANG_ENDPOINT_URL_BASELINE=https://... \
SGLANG_ENDPOINT_URL_KV_CACHE_QUANT=https://... \
SGLANG_ENDPOINT_URL_SPEC_DECODE=https://... \
make all
```

### Integrity verification after a stage run

```bash
python scripts/verify_run_integrity.py --run-dir "<results_run_dir>"
```
