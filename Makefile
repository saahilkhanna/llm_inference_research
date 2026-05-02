SHELL := /bin/bash

.PHONY: setup smoke smoke-from-scratch staged-backends single-workload all online shutdown clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke.sh

smoke-from-scratch:
	bash scripts/smoke_from_scratch.sh

staged-backends:
	bash scripts/run_backend_staged_sequence.sh

single-workload:
	bash scripts/run_single_workload_gsm8k.sh

all:
	bash scripts/run_all.sh

online:
	bash scripts/run_online_benchmarks.sh

shutdown:
	bash scripts/shutdown_endpoints.sh

clean:
	bash scripts/clean_results.sh
