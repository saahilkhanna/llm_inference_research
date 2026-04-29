from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .benchmark_selection import select_public_samples
from .compare_backends import compare_backends
from .config import load_config
from .custom_workload import build_custom_workload
from .endpoint_manager import EndpointInfo, apply_shutdown_mode, resolve_endpoints
from .generate_report import generate_final_report
from .grade_outputs import grade_outputs
from .parse_results import build_parsed_table
from .performance_runner import run_aiperf_profile
from .public_eval_runner import run_samples_for_backend
from .select_case_studies import select_case_studies
from .summarize_metrics import summarize_latency
from .standard_evaluator import run_standard_evaluator
from .tool_selection import generate_tool_selection_memo
from .utils import build_run_id, ensure_dir, system_manifest, to_plain_dict, utc_now_iso, write_json


def _build_run_dir(results_root: str, run_name: str) -> tuple[str, Path]:
    run_id = build_run_id(run_name)
    run_dir = Path(results_root) / run_id
    for rel in ["raw", "processed", "case_studies", "report", "logs"]:
        ensure_dir(run_dir / rel)
    return run_id, run_dir


def _write_manifest(
    run_dir: Path,
    run_id: str,
    mode: str,
    config_dict: dict,
    status: str,
    error: str = "",
    endpoints: dict | None = None,
) -> Path:
    payload = {
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "environment": system_manifest(),
        "config": config_dict,
        "endpoints": endpoints or {},
        "error": error,
    }
    path = run_dir / "manifest.json"
    write_json(path, payload)
    return path


def _run_pipeline(mode: str) -> int:
    smoke = mode == "smoke"
    config = load_config()
    run_id, run_dir = _build_run_dir(config.results_dir, config.run_name if not smoke else f"{config.run_name}_smoke")

    _write_manifest(run_dir, run_id, mode, to_plain_dict(config), "running")

    endpoints: dict[str, EndpointInfo] = {}
    try:
        generate_tool_selection_memo(config, run_dir)
        public_samples = select_public_samples(config, run_dir, smoke=smoke)
        custom_samples = build_custom_workload(config, run_dir, smoke=smoke)
        all_samples = public_samples + custom_samples
        if not all_samples:
            raise RuntimeError("No samples were selected. Enable at least one workload.")

        endpoints = resolve_endpoints(config, run_id=run_id, run_dir=run_dir)

        optimization_modes = ["baseline"] if smoke else config.optimization_modes
        repeats = 1 if smoke else config.repeats_per_condition

        total_conditions = len(optimization_modes) * repeats * len(config.engines)
        condition_index = 0

        for optimization_mode in optimization_modes:
            for repeat_index in range(1, repeats + 1):
                for engine in config.engines:
                    condition_index += 1
                    write_json(
                        run_dir / "logs" / "progress_status.json",
                        {
                            "stage": "running_samples",
                            "condition_index": condition_index,
                            "total_conditions": total_conditions,
                            "engine": engine,
                            "optimization_mode": optimization_mode,
                            "repeat_index": repeat_index,
                            "timestamp_utc": utc_now_iso(),
                        },
                    )
                    endpoint_url = config.endpoint_url_for(engine, optimization_mode, endpoints[engine].url)
                    run_samples_for_backend(
                        engine,
                        endpoint_url,
                        all_samples,
                        config,
                        run_dir,
                        optimization_mode=optimization_mode,
                        repeat_index=repeat_index,
                    )
                    write_json(
                        run_dir / "logs" / "progress_status.json",
                        {
                            "stage": "running_standard_evaluator",
                            "condition_index": condition_index,
                            "total_conditions": total_conditions,
                            "engine": engine,
                            "optimization_mode": optimization_mode,
                            "repeat_index": repeat_index,
                            "timestamp_utc": utc_now_iso(),
                        },
                    )
                    run_standard_evaluator(
                        config=config,
                        run_dir=run_dir,
                        backend=engine,
                        endpoint_url=endpoint_url,
                        optimization_mode=optimization_mode,
                        repeat_index=repeat_index,
                        smoke=smoke,
                    )
                    write_json(
                        run_dir / "logs" / "progress_status.json",
                        {
                            "stage": "condition_completed",
                            "condition_index": condition_index,
                            "total_conditions": total_conditions,
                            "engine": engine,
                            "optimization_mode": optimization_mode,
                            "repeat_index": repeat_index,
                            "timestamp_utc": utc_now_iso(),
                        },
                    )

        for engine in config.engines:
            run_aiperf_profile(engine, endpoints[engine].url, config, run_dir, smoke=smoke)

        parsed = build_parsed_table(run_dir)
        graded = grade_outputs(parsed)
        if not graded.empty:
            graded.to_csv(run_dir / "processed" / "all_samples_graded.csv", index=False)
        compare_backends(graded, run_dir)
        select_case_studies(run_dir, per_bucket=int(config.analysis.get("case_studies_per_bucket", 3)))
        summarize_latency(graded, run_dir)
        generate_final_report(config, run_dir, run_id)

        endpoint_manifest = {name: {"name": info.name, "url": info.url, "created": info.created} for name, info in endpoints.items()}
        _write_manifest(run_dir, run_id, mode, to_plain_dict(config), "completed", endpoints=endpoint_manifest)
    except Exception as exc:  # noqa: BLE001
        endpoint_manifest = {}
        if endpoints:
            endpoint_manifest = {
                name: {"name": info.name, "url": info.url, "created": info.created} for name, info in endpoints.items()
            }
        _write_manifest(
            run_dir,
            run_id,
            mode,
            to_plain_dict(config),
            "failed",
            error=f"{exc}\n{traceback.format_exc()}",
            endpoints=endpoint_manifest,
        )
        raise
    finally:
        endpoint_manifest = {name: {"name": info.name, "url": info.url, "created": info.created} for name, info in endpoints.items()}
        apply_shutdown_mode(config, endpoint_manifest, run_dir)

    print(f"Run complete: {run_dir}")
    return 0


def _run_shutdown() -> int:
    config = load_config()
    run_id, run_dir = _build_run_dir(config.results_dir, f"{config.run_name}_shutdown")
    endpoints = resolve_endpoints(config, run_id=run_id, run_dir=run_dir)
    endpoint_manifest = {name: {"name": info.name, "url": info.url, "created": info.created} for name, info in endpoints.items()}
    apply_shutdown_mode(config, endpoint_manifest, run_dir)
    _write_manifest(run_dir, run_id, "shutdown", to_plain_dict(config), "completed", endpoints=endpoint_manifest)
    print(f"Shutdown flow complete: {run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Black-box vLLM vs SGLang study runner")
    parser.add_argument("--mode", choices=["all", "smoke", "shutdown"], default="all")
    args = parser.parse_args()

    if args.mode == "shutdown":
        return _run_shutdown()
    return _run_pipeline(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
