# Run Flags and Execution Guide

This guide lists all runtime flags/env variables and practical command patterns to run the project for class research experiments.

## Quick Start Patterns

### 1) Smoke test (cheap validation)
```bash
RESULTS_DIR=real_results CREATE_ENDPOINTS=false BACKEND_API_STYLE=openai_chat make smoke
```

### 1b) GSM8K 7-endpoint smoke
```bash
make gsm8k-7-smoke
```
Runs all seven endpoint conditions with `LIMIT=1`, `PUBLIC_TASK_IDS=gsm8k_main`, custom workload disabled, standard eval disabled, and AIPerf disabled by default.

### 2) Full run (manual endpoints)
```bash
RESULTS_DIR=real_results CREATE_ENDPOINTS=false BACKEND_API_STYLE=openai_chat make all
```

### 2c) Online benchmarks only (no custom workload)
```bash
RESULTS_DIR=real_results \
CREATE_ENDPOINTS=false \
BACKEND_API_STYLE=openai_chat \
make online
```

### 2b) Professor-style matrix run (Control + vLLM + SGLang, 3 repeats)
```bash
ENGINES=llama_cpp,vllm,sglang \
OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode \
REPEATS_PER_CONDITION=3 \
LLAMA_CPP_ENDPOINT_URL=http://localhost:8080 \
RESULTS_DIR=real_results \
CREATE_ENDPOINTS=false \
BACKEND_API_STYLE=openai_chat \
make all
```
`llama_cpp` is treated as the minimal production baseline and only runs `baseline`; vLLM and SGLang run all configured optimization modes.

### 3) Full run (auto-create condition endpoints)
```bash
ENGINES=llama_cpp,vllm,sglang \
OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode \
LLAMA_CPP_MODEL_ID=<gguf-model-repo> \
RESULTS_DIR=real_results \
CREATE_ENDPOINTS=true \
make all
```
This resolves seven condition endpoints: `llama_cpp:baseline`, three vLLM modes, and three SGLang modes. If a condition URL is not supplied, the endpoint manager creates the matching managed endpoint. For llama.cpp managed creation, use a GGUF model repo via `LLAMA_CPP_MODEL_ID`.

### 4) Endpoint lifecycle only
```bash
make shutdown
```

---

## Env Flags Reference

Values are loaded from `.env` and can be overridden inline per command.

### Core authentication and model

- `HF_TOKEN`
  - HF access token used for protected endpoint calls and endpoint management.
  - Required when `CREATE_ENDPOINTS=true`.
- `HF_NAMESPACE`
  - HF user/org namespace for managed endpoint creation.
  - Optional if default account namespace is acceptable.
- `MODEL_ID`
  - Model string sent in chat/completion payloads and used for endpoint creation.
  - Must match what the endpoint expects to avoid model 404 errors.
- `FALLBACK_MODEL_ID`
  - Used only in endpoint auto-create path if `MODEL_ID` is gated/restricted.

### Endpoint routing

- `VLLM_ENDPOINT_URL`
  - URL for vLLM endpoint in manual mode.
- `SGLANG_ENDPOINT_URL`
  - URL for SGLang endpoint in manual mode.
- `VLLM_ENDPOINT_URL_BASELINE`, `VLLM_ENDPOINT_URL_KV_CACHE_QUANT`, `VLLM_ENDPOINT_URL_SPEC_DECODE`
  - Optional per-mode vLLM URLs. If set, these override `VLLM_ENDPOINT_URL` for that mode.
- `SGLANG_ENDPOINT_URL_BASELINE`, `SGLANG_ENDPOINT_URL_KV_CACHE_QUANT`, `SGLANG_ENDPOINT_URL_SPEC_DECODE`
  - Optional per-mode SGLang URLs. If set, these override `SGLANG_ENDPOINT_URL` for that mode.
- `LLAMA_CPP_MODEL_ID`, `VLLM_MODEL_ID`, `SGLANG_MODEL_ID`
  - Optional per-engine model repositories. For llama.cpp managed endpoints, use a GGUF model repo so Hugging Face selects the llama.cpp engine.
- `LLAMA_CPP_ENDPOINT_FRAMEWORK`, `VLLM_ENDPOINT_FRAMEWORK`, `SGLANG_ENDPOINT_FRAMEWORK`
  - `model.framework` value in the lower-level endpoint payload. Defaults to `pytorch`, matching the reference payload.
  - vLLM/SGLang engine selection is controlled by `model.image.vLLM` / `model.image.sGLang`.
- `ENDPOINT_ACCELERATOR`, `ENDPOINT_VENDOR`, `ENDPOINT_REGION`, `ENDPOINT_TYPE`, `ENDPOINT_INSTANCE_SIZE`, `ENDPOINT_INSTANCE_TYPE`
  - Managed endpoint creation settings.
- `TARGET_LLM_BASE_URL`
  - Convenience alias. If set while vLLM/SGLang URLs are empty, project uses this single URL for vLLM and SGLang and disables auto-create.
- `LLAMA_CPP_ENDPOINT_URL`
  - URL for a manual llama.cpp control endpoint, usually an OpenAI-compatible llama.cpp server.
  - Required when `ENGINES` includes `llama_cpp` only if `CREATE_ENDPOINTS=false`.
- `CREATE_ENDPOINTS`
  - `true`: create one endpoint per missing condition via HF APIs.
  - `false`: use provided endpoint URLs.
- `SHUTDOWN_MODE`
  - `pause | delete | scale_to_zero | none`
  - Applied at end of run.

### Run scope and generation

- `ENGINES`
  - Comma-separated list of engines to evaluate.
  - Supported: `llama_cpp`, `vllm`, `sglang`.
  - Example: `ENGINES=llama_cpp,vllm,sglang`
  - `llama_cpp` runs as a baseline-only control endpoint; vLLM and SGLang run the configured optimization modes.
- `OPTIMIZATION_MODES`
  - Comma-separated list of optimization conditions.
  - Supported: `baseline`, `kv_cache_quant`, `spec_decode`.
  - Example: `OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode`
  - With all three engines enabled, this produces seven endpoint conditions: one llama.cpp baseline plus three vLLM and three SGLang conditions.
- `REPEATS_PER_CONDITION`
  - Integer repeat count per engine x optimization condition.
  - Example: `REPEATS_PER_CONDITION=3`
- `RUN_BACKENDS_SEQUENTIALLY`
  - Keep `true` for predictable billing and cleaner logs.
- `LIMIT`
  - Full-run sample cap.
- `SMOKE_LIMIT`
  - Smoke sample cap.
- `TEMPERATURE`
- `TOP_P`
- `MAX_NEW_TOKENS`
  - Decoding controls applied to all calls.
- `REQUEST_TIMEOUT_SECONDS`
  - Per-request timeout.
- `BACKEND_API_STYLE`
  - `auto | openai_chat | tgi`
  - Set `openai_chat` for `/v1/chat/completions` style endpoints.
  - Set `tgi` for TGI text-generation style endpoints.
- `KV_CACHE_QUANT_MODE`
  - Label/hint value for KV cache quant mode (for example `fp8`, `int4`).
- `SPECULATIVE_DRAFT_MODEL`
  - Draft model hint string for speculative decoding trials.
- `SPECULATIVE_NUM_TOKENS`
  - Speculative token count hint.
- `VLLM_BASELINE_ARGS`, `VLLM_KV_CACHE_QUANT_ARGS`, `VLLM_SPEC_DECODE_ARGS`
  - Shell-style vLLM server args inserted into the managed endpoint `model.args` field.
  - Example: `VLLM_KV_CACHE_QUANT_ARGS="--gpu-memory-utilization 0.90 --max-model-len 8096 --kv-cache-dtype fp8"`
- `SGLANG_BASELINE_ARGS`, `SGLANG_KV_CACHE_QUANT_ARGS`, `SGLANG_SPEC_DECODE_ARGS`
  - Shell-style SGLang server args inserted into the managed endpoint `model.args` field.
  - Example: `SGLANG_KV_CACHE_QUANT_ARGS="--trust-remote-code --mem-fraction-static 0.90 --max-total-tokens 8096 --kv-cache-dtype fp8_e5m2"`
- `LLAMA_CPP_BASELINE_ARGS`
  - Optional shell-style args for the managed llama.cpp baseline endpoint.
- `APPLY_OPTIMIZATION_REQUEST_HINTS`
  - `true|false`.
  - If true, optimization hints are inserted into request payloads.
  - Keep false if your endpoint rejects unknown fields and configure optimization server-side instead.
- `STANDARD_EVAL_ENABLED`
  - Enables a standard external evaluator stage in the run loop.
- `STANDARD_EVALUATOR`
  - `lm_eval | none`
  - `lm_eval` runs lm-evaluation-harness and stores raw evaluator outputs under `raw/standard_evals/`.
- `STANDARD_EVAL_TASKS`
  - Comma-separated lm-eval task list.
  - Example: `gsm8k,hellaswag`
- `STANDARD_EVAL_RUN_ON_ALL_CONDITIONS`
  - If true, evaluator runs for every optimization/repeat condition.
  - If false, evaluator runs only for baseline repeat 1 (cost saver).
- `STANDARD_EVAL_LIMIT_OVERRIDE`
  - Optional cap for standard evaluator sample count (`0` disables override).
- `PUBLIC_TASK_IDS`
  - Optional comma-separated filter over `config/default.yaml` public task IDs.
  - Example: `PUBLIC_TASK_IDS=gsm8k_main`

### Workload toggles

- `PUBLIC_BENCHMARK_ENABLED`
  - Enable public benchmark sample set.
- `CUSTOM_WORKLOAD_ENABLED`
  - Enable custom Edge/IoT workload.
  - Set `false` to run online/public benchmarks only.
- `CUSTOM_WORKLOAD_SIZE`
  - Number of custom samples to generate.

### Output and budgeting

- `RESULTS_DIR`
  - Base output directory (for example `results` or `real_results`).
- `RUN_NAME`
  - Prefix used in run id folders.
- `MAX_ESTIMATED_COST_USD`
  - Budget guard metadata (tracked in run config/report context).

### Performance profiling

- `AIPERF_ENABLED`
  - Enables AIPerf profiling stage (if tool available).
- `AIPERF_SYNTHETIC_ENABLED`
  - Enables synthetic traffic options in AIPerf.

---

## Practical Run Recipes

### Manual endpoints (recommended for stable experiments)
Set in `.env`:
```env
CREATE_ENDPOINTS=false
BACKEND_API_STYLE=openai_chat
VLLM_ENDPOINT_URL=https://...
SGLANG_ENDPOINT_URL=https://...
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
RESULTS_DIR=real_results
```
Then:
```bash
make smoke
make all
```

### Auto-create endpoints (managed flow)
Set in `.env`:
```env
CREATE_ENDPOINTS=true
HF_TOKEN=...
HF_NAMESPACE=...
MODEL_ID=<model-you-can-access>
```
Then:
```bash
make smoke
make all
```

### Single-endpoint quick sanity run
If you only have one endpoint temporarily:
```bash
CREATE_ENDPOINTS=false \
VLLM_ENDPOINT_URL=https://same-endpoint \
SGLANG_ENDPOINT_URL=https://same-endpoint \
BACKEND_API_STYLE=openai_chat \
make smoke
```
Use only for wiring checks, not true backend comparison conclusions.

### True optimization ablation (recommended)
Use separate endpoint URLs per optimization mode so each condition is actually different server-side:
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
STANDARD_EVAL_ENABLED=true \
STANDARD_EVALUATOR=lm_eval \
STANDARD_EVAL_TASKS=gsm8k,hendrycks_math,mbpp \
STANDARD_EVAL_RUN_ON_ALL_CONDITIONS=true \
make all
```
The llama.cpp endpoint remains the baseline control in this matrix; only vLLM and SGLang use the per-mode optimization endpoints.

### Public benchmark focus with stronger online tasks
```bash
CUSTOM_WORKLOAD_ENABLED=false \
STANDARD_EVAL_ENABLED=true \
STANDARD_EVALUATOR=lm_eval \
STANDARD_EVAL_TASKS=gsm8k_platinum,hendrycks_math500,minerva_math,humaneval_instruct \
STANDARD_EVAL_RUN_ON_ALL_CONDITIONS=true \
STANDARD_EVAL_LIMIT_OVERRIDE=30 \
make online
```

---

## Recommended Experiment Matrix (for your optimization questions)

To study silent correctness issues and optimization effects, run separate conditions and compare run folders:

1. `baseline`
2. `kv_cache_quant`
3. `spec_decode`

For each condition:
- Keep prompts fixed (`LIMIT`, task set, custom workload size).
- Keep decoding params fixed (`TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`).
- Change only one backend optimization setting at a time.
- Run both backends under the same condition.

Use `RUN_NAME` per condition:
```bash
RUN_NAME=baseline RESULTS_DIR=real_results make all
RUN_NAME=kv_cache_quant RESULTS_DIR=real_results make all
RUN_NAME=spec_decode RESULTS_DIR=real_results make all
```

---

## Current Limitation About LightEval

Current code references LightEval in the tool-selection memo, but runtime evaluation is custom:
- sampling: `src/benchmark_selection.py`
- endpoint calls: `src/public_eval_runner.py`
- grading/comparison: `src/grade_outputs.py`, `src/compare_backends.py`

So the project can answer black-box behavior questions, but does **not** currently execute LightEval itself as the primary evaluator.
