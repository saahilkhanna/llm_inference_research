SHELL := /bin/bash

.PHONY: setup smoke all online shutdown clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke.sh

all:
	bash scripts/run_all.sh

online:
	bash scripts/run_online_benchmarks.sh

shutdown:
	bash scripts/shutdown_endpoints.sh

clean:
	bash scripts/clean_results.sh
