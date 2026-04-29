# Black-Box Inference Backend Failure Study

This project builds a reproducible, one-command research pipeline to compare **llama.cpp**, **vLLM**, and **SGLang** as black-box inference backends.

It answers:
1. How a minimal llama.cpp serving baseline differs from vLLM and SGLang in performance and failure behavior across public and custom workloads.
2. Whether optimization-oriented inference settings correlate with observable quality failures, especially for long-context and workload-specific prompts.

## What this project does

- Runs the same cached public and custom prompts against the configured backends.
- Saves raw request/response artifacts per sample.
- Grades outputs conservatively (`correct`, `wrong`, `unknown`).
- Produces per-sample comparison CSVs and failure buckets:
  - `both_correct`
  - `both_wrong`
  - `vllm_correct_sglang_wrong`
  - `sglang_correct_vllm_wrong`
  - `both_unknown`
  - `mixed_unknown`
- Summarizes latency and success metrics.
- Extracts representative case studies.
- Generates a final Markdown report.

## Claim boundaries

We treat llama.cpp, vLLM, and SGLang as black-box inference backends. Because the experiment uses managed endpoints and request-level logs, we focus on observable behavior rather than low-level GPU internals.

We do not claim that observed quality differences are definitively caused by KV-cache management, scheduling, or GPU memory behavior. Establishing that requires lower-level instrumentation and controlled ablations.

## Project layout

```text
.
├── README.md
├── Makefile
├── .gitignore
├── .env.example
├── requirements.txt
├── config/
│   └── default.yaml
├── scripts/
│   ├── run_all.sh
│   ├── run_smoke.sh
│   ├── setup_env.sh
│   ├── shutdown_endpoints.sh
│   └── clean_results.sh
├── src/
│   ├── config.py
│   ├── tool_selection.py
│   ├── benchmark_selection.py
│   ├── endpoint_manager.py
│   ├── public_eval_runner.py
│   ├── custom_workload.py
│   ├── performance_runner.py
│   ├── parse_results.py
│   ├── normalize_outputs.py
│   ├── grade_outputs.py
│   ├── compare_backends.py
│   ├── select_case_studies.py
│   ├── summarize_metrics.py
│   └── generate_report.py
├── data/
│   ├── public_samples/
│   └── custom_workloads/
├── results/
│   └── .gitkeep
└── report/
    └── final_summary_template.md
```

## Configuration

```bash
cp .env.example .env
```

Python requirement:
- Use Python `3.11+` (recommended `3.12`).
- `make setup` will fail fast if Python is too old.

Set key variables in `.env`:
- `HF_TOKEN`, `HF_NAMESPACE`, `MODEL_ID`
- `FALLBACK_MODEL_ID` (used automatically if `MODEL_ID` is access-restricted)
- `TARGET_LLM_BASE_URL` (optional single-endpoint alias for manual mode)
- `CREATE_ENDPOINTS` (recommended `true`)
- `VLLM_ENDPOINT_URL`, `SGLANG_ENDPOINT_URL` (only required if `CREATE_ENDPOINTS=false`)
- `LLAMA_CPP_MODEL_ID`, `VLLM_MODEL_ID`, `SGLANG_MODEL_ID` (optional per-engine model repos; llama.cpp should use a GGUF repo)
- `LLAMA_CPP_ENDPOINT_FRAMEWORK`, `VLLM_ENDPOINT_FRAMEWORK`, `SGLANG_ENDPOINT_FRAMEWORK` (`model.framework`, defaults to `pytorch`)
- `ENGINES`, `OPTIMIZATION_MODES`, `REPEATS_PER_CONDITION`
- `LLAMA_CPP_ENDPOINT_URL` (required for llama.cpp only when `CREATE_ENDPOINTS=false`)
- `KV_CACHE_QUANT_MODE`, `SPECULATIVE_DRAFT_MODEL`, `SPECULATIVE_NUM_TOKENS`
- `APPLY_OPTIMIZATION_REQUEST_HINTS` (inject optimization hints into request payloads)
- `STANDARD_EVAL_ENABLED`, `STANDARD_EVALUATOR`, `STANDARD_EVAL_TASKS`
- `STANDARD_EVAL_RUN_ON_ALL_CONDITIONS`, `STANDARD_EVAL_LIMIT_OVERRIDE`
- `CREATE_ENDPOINTS`, `SHUTDOWN_MODE`
- `RUN_BACKENDS_SEQUENTIALLY`
- `LIMIT`, `SMOKE_LIMIT`
- `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`
- `PUBLIC_BENCHMARK_ENABLED`, `CUSTOM_WORKLOAD_ENABLED`, `CUSTOM_WORKLOAD_SIZE`
- `RESULTS_DIR`, `RUN_NAME`, `MAX_ESTIMATED_COST_USD`

Endpoint behavior:
- Endpoint resolution is per condition, using keys like `vllm:baseline`, `vllm:kv_cache_quant`, and `sglang:spec_decode`.
- If `CREATE_ENDPOINTS=true`, the pipeline creates one managed Hugging Face endpoint per missing condition using the lower-level `/v2/endpoint/{namespace}` payload shape.
- vLLM/SGLang managed endpoints receive condition-specific `model.args` and engine image blocks (`vLLM` / `sGLang`) matching the Hugging Face endpoint configuration model.
- `llama_cpp` can be either a manual baseline endpoint via `LLAMA_CPP_ENDPOINT_URL` or a managed GGUF endpoint via `LLAMA_CPP_MODEL_ID`.
- If `CREATE_ENDPOINTS=false`, provide URLs for each configured backend or per-mode vLLM/SGLang URLs for true optimization ablations.
- If `TARGET_LLM_BASE_URL` is set and vLLM/SGLang URLs are empty, the pipeline uses that URL for vLLM and SGLang and automatically skips endpoint creation.
- With `ENGINES=llama_cpp,vllm,sglang` and `OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode`, the runner uses seven endpoint conditions: one llama.cpp baseline plus three vLLM and three SGLang conditions.
Configure server-side args with `VLLM_BASELINE_ARGS`, `VLLM_KV_CACHE_QUANT_ARGS`, `VLLM_SPEC_DECODE_ARGS`, `SGLANG_BASELINE_ARGS`, `SGLANG_KV_CACHE_QUANT_ARGS`, and `SGLANG_SPEC_DECODE_ARGS`.

You can change model, benchmark size, workload size, endpoint strategy, and shutdown behavior without editing code.

## Commands

```bash
make setup
make smoke
make all
make shutdown
```

- `make setup`: create/update virtual environment and install dependencies.
- `make smoke`: short low-cost run to validate wiring and artifacts.
- `make all`: full configured run.
- `make shutdown`: apply configured endpoint shutdown policy.

For complete run flags and command recipes, see `RUN_FLAGS.md`.

## Cost safety

- Smoke test first (`make smoke`).
- Backends run sequentially by default (`RUN_BACKENDS_SEQUENTIALLY=true`).
- Endpoints are paused by default after run (`SHUTDOWN_MODE=pause`).
- The manifest and report include runtime/cost context when available.

If endpoints are manually supplied and persistent, verify billing state in Hugging Face dashboard after runs.

## Where results are saved

Each run writes to `results/<RUN_ID>/`:
- `manifest.json`
- `tool_selection_memo.md`
- `raw/`
- `processed/public_per_sample_comparison.csv`
- `processed/custom_per_sample_comparison.csv`
- `processed/failure_bucket_summary.csv`
- `processed/engine_mode_summary.csv`
- `processed/latency_summary.csv`
- `case_studies/case_studies.csv`
- `case_studies/case_studies.md`
- `report/final_summary.md`
- `report/final_summary.txt`
- `logs/`

## Interpreting failure buckets

- `both_correct`: both backends answered correctly.
- `both_wrong`: both failed.
- `vllm_correct_sglang_wrong`: disagreement favoring vLLM.
- `sglang_correct_vllm_wrong`: disagreement favoring SGLang.
- `both_unknown`: no objective correctness signal for either.
- `mixed_unknown`: one side unknown and the other known.

These buckets support black-box failure characterization without overclaiming internal causes.
`engine_mode_summary.csv` and `latency_summary.csv` include all configured engines, including the llama.cpp baseline.
