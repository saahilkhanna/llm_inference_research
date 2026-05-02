# Black-Box Inference Backend Failure Study

This is our pipeline for comparing **llama.cpp**, [**vLLM**](https://github.com/vllm-project/vllm), and [**SGLang**](https://github.com/sgl-project/sglang) as black-box inference backends via HTTP.

**Documentation:** README.md (this file) is the main entry. Use **RUN_FLAGS.md** for environment variables, flags, and command recipes, and **RUN_WORKLOAD.md** for workload sizing and what each mode runs.

---

## How to execute yourself, we tried to make it easy! (fresh clone, credentials, one-command smoke)

Follow these steps:. **Use your own Hugging Face token.** We do not provide credentials.

### 1) Prerequisites on your machine

- **Python 3.11+** (3.12 recommended). 
- A **[Hugging Face](https://huggingface.co/join) account**.

### 2) Model access (example: Meta Llama)

The defaults assume an instruct Llama checkpoint such as [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). **You must open that model page while logged in and accept Meta’s license and request access** before our scripts can pull or deploy it. If you cannot use Llama, set MODEL_ID (and per-engine IDs if needed) in .env to another gated or open model you are authorized to use.

### 3) Hugging Face token

Create a token under **[Settings → Access Tokens](https://huggingface.co/settings/tokens)**. For managed **Inference Endpoints** (CREATE_ENDPOINTS=true), use a token that can provision endpoints in your namespace.

### 4) Configure and run the minimal smoke script

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, HF_NAMESPACE, MODEL_ID (and anything else your account needs).

chmod +x scripts/smoke_from_scratch.sh   # once per clone (already executable if you preserved git file modes)
./scripts/smoke_from_scratch.sh
# or: make smoke-from-scratch
```

What **scripts/smoke_from_scratch.sh** does:

1. Runs **scripts/setup_env.sh**, which creates **.venv** and pip install -r requirements.txt (installs **lm-eval** so the **lm_eval** CLI is available for full runs that enable **STANDARD_EVAL_ENABLED**).
2. Verifies **HF_TOKEN** and **HF_NAMESPACE** are set (parses .env with **python-dotenv**, avoiding fragile source .env on complex values).
3. Runs **python -m src.main --mode smoke** with **conservative overrides**: by default single **vLLM** engine, **baseline** optimization only, **STANDARD_EVAL_ENABLED=false** for this first wiring check. Smoke mode still respects **SMOKE_LIMIT** from .env.

**Disclaimer:** Our research runs used **staged multi-backend grids**, HumanEval and lm_eval, and larger limits; that is **more in-depth** than this smoke path. The script above is only the **shortest path** to prove the repo, Python env, HF auth, and endpoint path work end-to-end.

Optional environment overrides for the smoke script (all optional):

| Variable | Default in script | Meaning |
|----------|-------------------|---------|
| SMOKE_ENGINES | vllm | Engines to include in smoke |
| SMOKE_OPTIMIZATION_MODES | baseline | Optimization modes |
| SMOKE_STANDARD_EVAL_ENABLED | false | Set true to exercise **lm_eval** during smoke (slower; needs tasks configured) |
| SMOKE_CREATE_ENDPOINTS | true | Set false only if you already have a reachable endpoint URL and configure manual URL vars (see RUN_FLAGS.md) |

---

## Experimental configuration, open-source stack, code size, and datasets

This repository implements an **experimental measurement** workflow: replay fixed workloads against live inference endpoints and record latency, token estimates, HTTP outcomes, and graded correctness (plus optional external eval).

### Experimental configurations

- **Backends compared:** llama_cpp, vllm, sglang via ENGINES.
- **Optimization conditions:** baseline, kv_cache_quant, spec_decode via OPTIMIZATION_MODES. Llama.cpp runs **baseline only**; vLLM and SGLang run each configured mode. Server-side behavior is set through the per-engine VLLM and SGLANG argument blocks in env (see src/config.py) and KV-cache / speculative-decoding env knobs.
- **Serving topology:** defaults target **Hugging Face Inference Endpoints** (CREATE_ENDPOINTS=true), resolving one URL per (engine, optimization_mode). Manual URLs apply when CREATE_ENDPOINTS=false.
- **Generation/decoding:** TEMPERATURE, TOP_P, MAX_NEW_TOKENS, REQUEST_TIMEOUT_SECONDS; workload size via LIMIT / SMOKE_LIMIT; optional STANDARD_EVAL_* variables for harness tasks.
- **Artifacts:** each run writes **manifest.json**, **logs/**, **raw/**, **processed/**, and optionally **report/** / **case_studies/**.

### Open-source software: inference servers (what actually serves the model)

These are the **open-source inference stacks** we deploy behind HF endpoints (container images and CLI flags are wired in src/config.py and src/endpoint_manager.py):

| Stack | Role | Reference |
|-------|------|-----------|
| **vLLM** | High-throughput LLM serving with OpenAI-compatible APIs | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **SGLang** | Structured generation oriented serving runtime | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| **llama.cpp** | GGUF / CPU-GPU llama.cpp HTTP server path for a strong baseline | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |

### Open-source software: evaluation (**lm-eval** / lm-evaluation-harness)

For standardized tasks we invoke the **lm_eval** CLI from **[EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** (lm-eval on PyPI, declared in **requirements.txt**). Configuration lives in **src/standard_evaluator.py** (STANDARD_EVAL_ENABLED, STANDARD_EVAL_TASKS, HumanEval code-eval flags, etc.).

### Open-source software: Python runtime and infra clients

Pinned in **[requirements.txt](requirements.txt)**: [python-dotenv](https://github.com/theskumar/python-dotenv), [PyYAML](https://pyyaml.org/), [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [requests](https://requests.readthedocs.io/), [datasets](https://github.com/huggingface/datasets), [huggingface_hub](https://github.com/huggingface/huggingface_hub), [tqdm](https://github.com/tqdm/tqdm), **lm-eval**, [tenacity](https://github.com/jd/tenacity), [matplotlib](https://matplotlib.org/), etc. API provisioning uses **[Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints)** (src/endpoint_manager.py).

### Code size (this repo, approximate)

- **src/**: ~3.1k lines of Python  
- **scripts/**: ~260+ lines (shell + helper Python), including **smoke_from_scratch.sh**  

*(Counts exclude .venv and run output folders.)*

### Benchmark and simulation datasets

Defined in [config/default.yaml](config/default.yaml) (public_tasks):

| Task id (filter) | Dataset | Split (as configured) |
|------------------|---------|------------------------|
| humaneval_coding | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | test |
| gsm8k_main | [gsm8k](https://huggingface.co/datasets/gsm8k) (subset: main) | test |
| competition_math | [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) | test |
| longbench_v2_slice | [THUDM/LongBench-v2](https://huggingface.co/datasets/THUDM/LongBench-v2) | train (bounded slice in code) |

Optional **synthetic custom** prompts when CUSTOM_WORKLOAD_ENABLED=true (custom_workload in config/default.yaml). **lm_eval** tasks (e.g. humaneval_instruct) run when STANDARD_EVAL_ENABLED=true.

---

## Research questions

1. How does a minimal llama.cpp baseline differ from vLLM and SGLang in performance and failure behavior across workloads?
2. Do optimization-oriented inference settings correlate with observable quality failures (including long-context or workload-specific prompts)?

## What this project does
- Runs the workload prompts against the configured backends.
- Saves raw request/response artifacts per sample.
- Grades outputs conservatively (correct, wrong, unknown).
- Produces per-sample comparison CSVs and failure buckets:
  - both_correct
  - both_wrong
  - vllm_correct_sglang_wrong
  - sglang_correct_vllm_wrong
  - both_unknown
  - mixed_unknown
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
├── RUN_FLAGS.md
├── RUN_WORKLOAD.md
├── Makefile
├── .gitignore
├── .env.example
├── requirements.txt
├── config/
│   └── default.yaml
├── scripts/
│   ├── smoke_from_scratch.sh
│   ├── run_all.sh
│   ├── run_smoke.sh
│   ├── setup_env.sh
│   ├── shutdown_endpoints.sh
│   └── clean_results.sh
├── src/
│   ├── main.py
│   ├── config.py
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
│   ├── standard_evaluator.py
│   └── generate_report.py
├── data/
│   ├── public_samples/
│   └── custom_workloads/
├── results/
│   └── .gitkeep
└── report/
    └── final_summary_template.md   # required by generate_report if you run the Markdown report step
```

## Configuration

```bash
cp .env.example .env
```

Python requirement:
- Use Python 3.11+ (recommended 3.12).
- make setup will fail fast if Python is too old.

Set key variables in .env:
- HF_TOKEN, HF_NAMESPACE, MODEL_ID
- FALLBACK_MODEL_ID (used automatically if MODEL_ID is access-restricted)
- TARGET_LLM_BASE_URL (optional single-endpoint alias for manual mode)
- CREATE_ENDPOINTS (recommended true)
- VLLM_ENDPOINT_URL, SGLANG_ENDPOINT_URL (only required if CREATE_ENDPOINTS=false)
- LLAMA_CPP_MODEL_ID, VLLM_MODEL_ID, SGLANG_MODEL_ID (optional per-engine model repos; llama.cpp should use a GGUF repo)
- LLAMA_CPP_ENDPOINT_FRAMEWORK, VLLM_ENDPOINT_FRAMEWORK, SGLANG_ENDPOINT_FRAMEWORK (model.framework, defaults to pytorch)
- ENGINES, OPTIMIZATION_MODES, REPEATS_PER_CONDITION
- LLAMA_CPP_ENDPOINT_URL (required for llama.cpp only when CREATE_ENDPOINTS=false)
- KV_CACHE_QUANT_MODE, SPECULATIVE_DRAFT_MODEL, SPECULATIVE_NUM_TOKENS
- APPLY_OPTIMIZATION_REQUEST_HINTS (inject optimization hints into request payloads)
- STANDARD_EVAL_ENABLED, STANDARD_EVALUATOR, STANDARD_EVAL_TASKS
- STANDARD_EVAL_RUN_ON_ALL_CONDITIONS, STANDARD_EVAL_LIMIT_OVERRIDE
- CREATE_ENDPOINTS, SHUTDOWN_MODE
- RUN_BACKENDS_SEQUENTIALLY
- LIMIT, SMOKE_LIMIT
- TEMPERATURE, TOP_P, MAX_NEW_TOKENS
- PUBLIC_BENCHMARK_ENABLED, CUSTOM_WORKLOAD_ENABLED, CUSTOM_WORKLOAD_SIZE
- RESULTS_DIR, RUN_NAME, MAX_ESTIMATED_COST_USD

Endpoint behavior:
- Endpoint resolution is per condition, using keys like vllm:baseline, vllm:kv_cache_quant, and sglang:spec_decode.
- If CREATE_ENDPOINTS=true, the pipeline creates one managed Hugging Face endpoint per missing condition using the lower-level /v2/endpoint/{namespace} payload shape.
- vLLM/SGLang managed endpoints receive condition-specific model.args and engine image blocks (vLLM / sGLang) matching the Hugging Face endpoint configuration model.
- llama_cpp can be either a manual baseline endpoint via LLAMA_CPP_ENDPOINT_URL or a managed GGUF endpoint via LLAMA_CPP_MODEL_ID.
- If CREATE_ENDPOINTS=false, provide URLs for each configured backend or per-mode vLLM/SGLang URLs for true optimization ablations.
- If TARGET_LLM_BASE_URL is set and vLLM/SGLang URLs are empty, the pipeline uses that URL for vLLM and SGLang and automatically skips endpoint creation.
- With ENGINES=llama_cpp,vllm,sglang and OPTIMIZATION_MODES=baseline,kv_cache_quant,spec_decode, the runner uses seven endpoint conditions: one llama.cpp baseline plus three vLLM and three SGLang conditions.
Configure server-side args with VLLM_BASELINE_ARGS, VLLM_KV_CACHE_QUANT_ARGS, VLLM_SPEC_DECODE_ARGS, SGLANG_BASELINE_ARGS, SGLANG_KV_CACHE_QUANT_ARGS, and SGLANG_SPEC_DECODE_ARGS.

You can change model, benchmark size, workload size, endpoint strategy, and shutdown behavior without editing code.

## Commands

```bash
make setup
make smoke-from-scratch   # first-time minimal smoke (see "Start here")
make smoke
make all
make shutdown
```

- make setup: create/update virtual environment and install dependencies.
- make smoke-from-scratch: run **scripts/smoke_from_scratch.sh** (venv + dependency check + minimal smoke).
- make smoke: short low-cost run using your current .env without the smoke-script overrides.
- make all: full configured run.
- make shutdown: apply configured endpoint shutdown policy.

For complete run flags and command recipes, see RUN_FLAGS.md.

## We wanted to ensure results before we ran longer more in-depth runs
- Smoke test first (make smoke).
- Backends run sequentially by default (RUN_BACKENDS_SEQUENTIALLY=true).
- Endpoints are paused by default after run (SHUTDOWN_MODE=pause).

If endpoints are manually supplied and persistent, verify billing state in Hugging Face dashboard after runs.

## Where results are saved
Each run writes under <RESULTS_DIR>/<RUN_ID>/ (default RESULTS_DIR is results; override per run if needed):

- manifest.json
- raw/ (responses, optional standard_evals/ when lm_eval is enabled)
- processed/, including all_samples_graded.csv, comparison CSVs, summaries
- case_studies/, when comparison tables are non-empty
- report/, final_summary.md / final_summary.txt when report generation succeeds
- logs/, progress and endpoint resolution JSONL

logs/progress_status.json and related JSONL files update during the run so you can see current condition progress.

## Interpreting failure buckets

- both_correct: both backends answered correctly.
- both_wrong: both failed.
- vllm_correct_sglang_wrong: disagreement favoring vLLM.
- sglang_correct_vllm_wrong: disagreement favoring SGLang.
- both_unknown: no objective correctness signal for either.
- mixed_unknown: one side unknown and the other known.

These buckets support black-box failure characterization without overclaiming internal causes.
engine_mode_summary.csv and latency_summary.csv include all configured engines, including the llama.cpp baseline.

