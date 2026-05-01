SHELL := /bin/bash

.PHONY: setup smoke staged-backends single-workload single-workload-longbench reaggregate-overlap discrepancy-report optimization-regression-report data-collection all online shutdown clean

setup:
	bash scripts/setup_env.sh

smoke:
	bash scripts/run_smoke.sh

staged-backends:
	bash scripts/run_backend_staged_sequence.sh

single-workload:
	bash scripts/run_single_workload_gsm8k.sh

single-workload-longbench:
	bash scripts/run_single_workload_longbench.sh

reaggregate-overlap:
	source .venv/bin/activate && python scripts/reaggregate_overlap_samples.py

discrepancy-report:
	source .venv/bin/activate && python scripts/backend_discrepancy_report.py

optimization-regression-report:
	source .venv/bin/activate && python scripts/optimization_regression_report.py

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
