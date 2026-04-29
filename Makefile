SHELL := /bin/bash

.PHONY: setup smoke staged-backends data-collection all online shutdown clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke.sh

staged-backends:
	bash scripts/run_backend_staged_sequence.sh

data-collection:
	source .venv/bin/activate && python scripts/build_data_collection.py --results-dir real_results/USED_RESULTS --output-dir final_summary/data_collection

all:
	bash scripts/run_all.sh

online:
	bash scripts/run_online_benchmarks.sh

shutdown:
	bash scripts/shutdown_endpoints.sh

clean:
	bash scripts/clean_results.sh
