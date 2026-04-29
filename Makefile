SHELL := /bin/bash

.PHONY: setup smoke gsm8k-7-smoke all online shutdown clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke.sh

gsm8k-7-smoke:
	bash scripts/run_gsm8k_7_endpoint_smoke.sh

all:
	bash scripts/run_all.sh

online:
	bash scripts/run_online_benchmarks.sh

shutdown:
	bash scripts/shutdown_endpoints.sh

clean:
	bash scripts/clean_results.sh
