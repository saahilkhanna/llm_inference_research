# How to run a workload

This document explains how to execute the benchmarking pipeline against your backends. For every environment variable and flag, see **`RUN_FLAGS.md`**. For project goals and layout, see **`README.md`**.

## 1. One-time setup

```bash
cp .env.example .env   # edit .env before real runs (token, namespace, models)
make setup             # Python 3.11+ recommended; creates .venv and installs deps
```

## 2. How workloads are dispatched

The runner loads configuration from `.env` (see `src/config.py`). It builds a **list of conditions**: each `(engine, optimization_mode)` pair gets its own Hugging Face Inference Endpoint URL (when using managed endpoints) or a URL you supply (manual mode). For each condition it runs the same **sample workloads**—public benchmarks and/or synthetic custom prompts—that you enabled in the environment.

**Engines:** `ENGINES` (comma-separated): `llama_cpp`, `vllm`, `sglang`.

**Optimization modes:** `OPTIMIZATION_MODES`: `baseline`, `kv_cache_quant`, `spec_decode`.  
Llama.cpp is only evaluated at **`baseline`**; vLLM and SGLang are evaluated at **every** configured mode.

**Managed versus manual endpoints:**

| Approach | What you configure | Typical use |
|---------|---------------------|-------------|
| **Managed** (`CREATE_ENDPOINTS=true`) | `HF_TOKEN`, `HF_NAMESPACE`, model IDs, endpoint hardware env vars (`ENDPOINT_*`, engine images, args). No per-condition URLs required. | Let the pipeline create/update endpoints and discover URLs automatically. |
| **Manual** (`CREATE_ENDPOINTS=false`) | Base URLs (`VLLM_ENDPOINT_URL`, `SGLANG_ENDPOINT_URL`, optionally per-mode overrides like `VLLM_ENDPOINT_URL_KV_CACHE_QUANT`, `LLAMA_CPP_ENDPOINT_URL`, etc.). | All servers already running; compare fixed deployments. |

With managed endpoints you do **not** paste seven URLs: the resolver holds one URL per `(engine, optimization_mode)` after creation.

## 3. CLI modes

`python -m src.main` accepts **`--mode`**:

| Mode | Effect |
|------|--------|
| **`all`** | Uses full `ENGINES`, `OPTIMIZATION_MODES`, `REPEATS_PER_CONDITION`. Full pipeline: samples → optional standard eval → metrics → comparison → report. |
| **`smoke`** | Shrinks scopes (e.g. `OPTIMIZATION_MODES` collapses to `baseline`) for a wiring check. Lowest cost sanity run. |
| **`shutdown`** | Applies `SHUTDOWN_MODE` to endpoints (pause/delete/etc.) using current config—not a workload benchmark run. |

## 4. Common commands

Use these from the repository root after `make setup` and activating the venv (the scripts below source it when needed).

```bash
# Quick sanity check (smoke scope)
make smoke
# Equivalent: bash scripts/run_smoke.sh

# Full run per your current .env
make all
# Equivalent: bash scripts/run_all.sh

# GSM8k public task, smoke-sized limits, all engines and three optimization modes (seven endpoint conditions); creates managed endpoints unless disabled
make gsm8k-7-smoke
# Equivalent: bash scripts/run_gsm8k_7_endpoint_smoke.sh

# Apply shutdown policy to endpoints (pause is common)
make shutdown
# Equivalent to running main in shutdown mode; see scripts/shutdown_endpoints.sh
```

For ad-hoc overrides without editing `.env`:

```bash
ENGINES=vllm,sglang OPTIMIZATION_MODES=baseline PUBLIC_TASK_IDS=gsm8k_main LIMIT=10 \
  python -m src.main --mode all
```

## 5. What gets run on each backend

Workload content is driven by booleans such as **`PUBLIC_BENCHMARK_ENABLED`**, **`CUSTOM_WORKLOAD_ENABLED`**, and **`STANDARD_EVAL_ENABLED`**, plus task IDs like **`PUBLIC_TASK_IDS`**, **`CUSTOM_WORKLOAD_SIZE`**, and **`LIMIT`** / **`SMOKE_LIMIT`** (see `RUN_FLAGS.md`). Defaults can be overwritten per run via environment variables exactly as shown above.

## 6. Where results go

Each run creates a subdirectory under **`RESULTS_DIR`** (default `results` or overridden, e.g. `real_results` in the GSM8K smoke script):

```text
<RESULTS_DIR>/<RUN_ID>/
├── manifest.json
├── raw/
├── processed/
├── report/
├── logs/
└── …
```

`RUN_ID` is derived from a timestamp plus your **`RUN_NAME`**. Inspect `manifest.json` for resolved config and endpoint metadata.

## 7. Cost and safety habits

Run **`make smoke`** or **`make gsm8k-7-smoke`** before expensive full grids. Prefer **`SHUTDOWN_MODE=pause`** after runs so GPUs do not sit billing (see **`SHUTDOWN_MODE`** in `RUN_FLAGS.md`). **`MAX_ESTIMATED_COST_USD`** appears in manifests for bookkeeping; tighten it intentionally if your workflow uses it as a reminder. Review `RUN_FLAGS.md` for parallel provisioning and endpoint tuning.
