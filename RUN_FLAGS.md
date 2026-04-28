# Run Flags and Execution Guide

This guide lists all runtime flags/env variables and practical command patterns to run the project for class research experiments.

## Quick Start Patterns

### 1) Smoke test (cheap validation)
```bash
RESULTS_DIR=real_results CREATE_ENDPOINTS=false BACKEND_API_STYLE=openai_chat make smoke
```

### 2) Full run (manual endpoints)
```bash
RESULTS_DIR=real_results CREATE_ENDPOINTS=false BACKEND_API_STYLE=openai_chat make all
```

### 2b) Professor-style matrix run (Control + vLLM + SGLang, 3 repeats)
```bash
ENGINES=llama_cpp,vllm,sglang \
OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode \
REPEATS_PER_CONDITION=3 \
RESULTS_DIR=real_results \
CREATE_ENDPOINTS=false \
BACKEND_API_STYLE=openai_chat \
make all
```

### 3) Full run (auto-create HF endpoints)
```bash
RESULTS_DIR=real_results CREATE_ENDPOINTS=true make all
```

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
- `TARGET_LLM_BASE_URL`
  - Convenience alias. If set while both backend URLs are empty, project uses this single URL for both backends and disables auto-create.
- `CREATE_ENDPOINTS`
  - `true`: create/reuse endpoints via HF APIs.
  - `false`: use provided endpoint URLs.
- `SHUTDOWN_MODE`
  - `pause | delete | scale_to_zero | none`
  - Applied at end of run.

### Run scope and generation

- `ENGINES`
  - Comma-separated list of engines to evaluate.
  - Supported: `llama_cpp`, `vllm`, `sglang`.
  - Example: `ENGINES=llama_cpp,vllm,sglang`
- `OPTIMIZATION_MODES`
  - Comma-separated list of optimization conditions.
  - Supported: `baseline`, `kv_cache_quant`, `spec_decode`.
  - Example: `OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode`
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
- `LLAMA_CPP_ENDPOINT_URL`
  - Endpoint URL for the control group (`llama_cpp`) when enabled.
- `KV_CACHE_QUANT_MODE`
  - Label/hint value for KV cache quant mode (for example `fp8`, `int4`).
- `SPECULATIVE_DRAFT_MODEL`
  - Draft model hint string for speculative decoding trials.
- `SPECULATIVE_NUM_TOKENS`
  - Speculative token count hint.
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

### Workload toggles

- `PUBLIC_BENCHMARK_ENABLED`
  - Enable public benchmark sample set.
- `CUSTOM_WORKLOAD_ENABLED`
  - Enable custom Edge/IoT workload.
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

---

## Recommended Experiment Matrix (for your optimization questions)

To study silent correctness issues and optimization effects, run separate conditions and compare run folders:

1. `baseline`
2. `kv_quant`
3. `spec_decode`
4. `attention_opt`

For each condition:
- Keep prompts fixed (`LIMIT`, task set, custom workload size).
- Keep decoding params fixed (`TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`).
- Change only one backend optimization setting at a time.
- Run both backends under the same condition.

Use `RUN_NAME` per condition:
```bash
RUN_NAME=baseline RESULTS_DIR=real_results make all
RUN_NAME=kv_quant RESULTS_DIR=real_results make all
RUN_NAME=spec_decode RESULTS_DIR=real_results make all
RUN_NAME=attention_opt RESULTS_DIR=real_results make all
```

---

## Current Limitation About LightEval

Current code references LightEval in the tool-selection memo, but runtime evaluation is custom:
- sampling: `src/benchmark_selection.py`
- endpoint calls: `src/public_eval_runner.py`
- grading/comparison: `src/grade_outputs.py`, `src/compare_backends.py`

So the project can answer black-box behavior questions, but does **not** currently execute LightEval itself as the primary evaluator.
