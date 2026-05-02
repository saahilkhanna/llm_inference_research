#!/usr/bin/env bash
# Minimal wiring smoke test: creates .venv, installs deps (including lm-eval), runs a tiny pipeline smoke.
# You must supply your own Hugging Face credentials and model access (see README "Start here").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "== Black-box inference backends: smoke_from_scratch =="
echo ""
echo "Prerequisites (you do this once outside this script):"
echo "  1) Hugging Face account: https://huggingface.co/join"
echo "  2) Accept the license for your chosen MODEL_ID (e.g. Meta Llama): open the model page on HF and request access."
echo "  3) Create an access token: https://huggingface.co/settings/tokens"
echo "     (needs rights to use Inference Endpoints if CREATE_ENDPOINTS=true)."
echo ""

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo ""
    echo "NEXT: Edit .env and set at minimum:"
    echo "  HF_TOKEN=<your token>"
    echo "  HF_NAMESPACE=<your HF username or org used for Inference Endpoints>"
    echo "  MODEL_ID=<a model you have access to, e.g. meta-llama/Llama-3.2-3B-Instruct>"
    echo ""
    echo "Then re-run:"
    echo "  bash scripts/smoke_from_scratch.sh"
    exit 1
  else
    echo "Missing .env.example — cannot bootstrap .env"
    exit 1
  fi
fi

echo "== Step 1: Python venv + pip dependencies (includes lm-eval / lm_eval CLI) =="
bash "${ROOT}/scripts/setup_env.sh"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

python - <<'PY'
from pathlib import Path
from dotenv import dotenv_values

vals = dotenv_values(Path(".env"))
for key in ("HF_TOKEN", "HF_NAMESPACE"):
    if not (vals.get(key) or "").strip():
        raise SystemExit(f"ERROR: {key} is missing or empty in .env — see README.")
print("OK: HF_TOKEN and HF_NAMESPACE are set (length hidden).")
PY

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "ERROR: lm_eval CLI not found after pip install. Check requirements.txt contains lm-eval."
  exit 1
fi
echo "OK: lm_eval is available ($(command -v lm_eval))"

echo ""
echo "== Step 2: Minimal smoke run (single vLLM baseline, reduced scope) =="
echo "    This may provision ONE managed endpoint — HF may bill GPU time."
echo ""

# Overrides for cost/risk: one engine, baseline only, no lm_eval by default in smoke, tiny workload slice.
export ENGINES="${SMOKE_ENGINES:-vllm}"
export OPTIMIZATION_MODES="${SMOKE_OPTIMIZATION_MODES:-baseline}"
export CREATE_ENDPOINTS="${SMOKE_CREATE_ENDPOINTS:-true}"
export STANDARD_EVAL_ENABLED="${SMOKE_STANDARD_EVAL_ENABLED:-false}"
export PUBLIC_BENCHMARK_ENABLED="${SMOKE_PUBLIC_BENCHMARK_ENABLED:-true}"
export CUSTOM_WORKLOAD_ENABLED="${SMOKE_CUSTOM_WORKLOAD_ENABLED:-false}"
export PUBLIC_TASK_IDS="${SMOKE_PUBLIC_TASK_IDS:-gsm8k_main}"
export RUN_BACKENDS_SEQUENTIALLY="${SMOKE_RUN_BACKENDS_SEQUENTIALLY:-true}"
export RESULTS_DIR="${SMOKE_RESULTS_DIR:-results}"
export RUN_NAME="${SMOKE_RUN_NAME:-smoke_from_scratch}"

python -m src.main --mode smoke

echo ""
echo "Smoke finished. New run directory: look under ${RESULTS_DIR}/ for a folder named like *_${RUN_NAME}_smoke"
echo "(Smoke mode uses baseline-only optimizations and SMOKE_LIMIT-sized samples.)"
