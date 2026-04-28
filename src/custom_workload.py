from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .utils import ensure_dir, write_jsonl


def _short_sample(i: int) -> dict:
    a = i + 2
    b = i + 5
    return {
        "sample_id": f"custom_short_{i}",
        "workload": "custom",
        "task_id": "custom_short_arithmetic",
        "workload_class": "short",
        "prompt": f"What is {a} + {b}? Return only the number.",
        "expected_answer": str(a + b),
        "grading_type": "numeric",
        "source": "custom",
    }


def _medium_sample(i: int) -> dict:
    base = 100 + i
    prompt = (
        "An IoT edge device reports the following values.\n"
        f"temperature={base}, correction=7, offset=-3.\n"
        "Compute temperature + correction + offset. Return only the final number."
    )
    return {
        "sample_id": f"custom_medium_{i}",
        "workload": "custom",
        "task_id": "custom_medium_calc",
        "workload_class": "medium",
        "prompt": prompt,
        "expected_answer": str(base + 7 - 3),
        "grading_type": "numeric",
        "source": "custom",
    }


def _long_sample(i: int, min_chars: int) -> dict:
    sensor_id = f"edge-{i:04d}"
    expected = f"{30000 + i}"
    filler_sentence = "Telemetry chunk: voltage stable, humidity nominal, packet acknowledged. "
    filler = (filler_sentence * ((min_chars // len(filler_sentence)) + 3))[:min_chars]
    prompt = (
        "Read the long operations log and return the firmware code for the target sensor.\n\n"
        f"{filler}\n\n"
        f"Key-value entry: sensor={sensor_id}; firmware_code={expected}\n\n"
        f"Question: What is the firmware_code for sensor {sensor_id}? Return only the numeric code."
    )
    return {
        "sample_id": f"custom_long_{i}",
        "workload": "custom",
        "task_id": "custom_long_context_retrieval",
        "workload_class": "long",
        "prompt": prompt,
        "expected_answer": expected,
        "grading_type": "numeric",
        "source": "custom",
    }


def build_custom_workload(config: AppConfig, run_dir: Path, smoke: bool) -> list[dict]:
    if not config.custom_workload_enabled:
        return []

    total = min(config.custom_workload_size, config.effective_limit(smoke))
    if total <= 0:
        return []

    short_n = max(1, int(total * config.custom_workload.get("short_ratio", 0.33)))
    medium_n = max(1, int(total * config.custom_workload.get("medium_ratio", 0.33)))
    long_n = max(1, total - short_n - medium_n)

    min_chars = int(config.custom_workload.get("long_context_min_chars", 9000))

    samples: list[dict] = []
    samples.extend(_short_sample(i) for i in range(short_n))
    samples.extend(_medium_sample(i) for i in range(medium_n))
    samples.extend(_long_sample(i, min_chars=min_chars) for i in range(long_n))
    samples = samples[:total]

    out_path = run_dir / "raw" / "custom_samples.jsonl"
    ensure_dir(out_path.parent)
    write_jsonl(out_path, samples)
    return samples
