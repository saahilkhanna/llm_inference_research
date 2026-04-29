from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import ensure_dir, write_jsonl


def _build_mmlu_prompt(question: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    rendered_choices = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(min(4, len(choices))))
    return (
        "Answer the multiple-choice question. "
        "Return only one letter from A, B, C, or D.\n\n"
        f"Question: {question}\n{rendered_choices}\n\nAnswer:"
    )


def _build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the problem and provide the final numeric answer on the last line in the form "
        "'Answer: <number>'.\n\n"
        f"Problem: {question}\n\nAnswer:"
    )


def _extract_gsm8k_target(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def _extract_math_boxed(solution: str) -> str:
    # Many MATH references use \\boxed{...} as final answer.
    match = re.findall(r"\\boxed\{([^}]*)\}", solution or "")
    if match:
        return match[-1].strip()
    return solution.strip().split("\n")[-1].strip()


def _build_humaneval_prompt(prompt: str) -> str:
    return (
        "Complete the Python function body. Return only valid Python code.\n\n"
        f"{prompt}\n"
    )


def _make_long_context_public_style(idx: int) -> dict[str, Any]:
    random.seed(idx)
    needle = f"sensor-{idx:04d}"
    value = f"{7000 + idx}"
    filler = " ".join(["edge telemetry packet"] * 2000)
    prompt = (
        "Read the long context and extract the calibration code for the requested sensor.\n\n"
        f"Context:\n{filler}\n\n"
        f"Lookup Table: {needle} -> {value}\n\n"
        "Question: What is the calibration code for "
        f"{needle}? Return only the numeric code."
    )
    return {
        "sample_id": f"public_long_context_{idx}",
        "workload": "public",
        "task_id": "public_long_context_slice",
        "workload_class": "long",
        "prompt": prompt,
        "expected_answer": value,
        "grading_type": "numeric",
        "source": "public-benchmark-style",
    }


def select_public_samples(config: AppConfig, run_dir: Path, smoke: bool) -> list[dict[str, Any]]:
    if not config.public_benchmark_enabled:
        return []

    limit = config.effective_limit(smoke)
    public_tasks = config.public_tasks
    if config.public_task_ids:
        allowed_task_ids = set(config.public_task_ids)
        public_tasks = [task for task in public_tasks if task.get("id") in allowed_task_ids]
    target_per_task = max(1, limit // max(1, len(public_tasks)))
    samples: list[dict[str, Any]] = []

    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        load_dataset = None

    for task in public_tasks:
        task_id = task.get("id", "unknown_task")
        dataset_name = task.get("dataset", "")
        subset = task.get("subset") or None
        split = task.get("split", "test")
        grading = task.get("grading", "unknown")
        workload_class = task.get("workload_class", "short")

        if task_id.startswith("gsm8k"):
            try:
                if load_dataset is None:
                    raise RuntimeError("datasets package unavailable")
                ds = load_dataset(dataset_name, subset, split=split)
                for i, row in enumerate(ds.select(range(min(target_per_task, len(ds))))):
                    samples.append(
                        {
                            "sample_id": f"{task_id}_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": workload_class,
                            "prompt": _build_gsm8k_prompt(row["question"]),
                            "expected_answer": _extract_gsm8k_target(row["answer"]),
                            "grading_type": grading,
                            "source": dataset_name,
                        }
                    )
            except Exception:  # noqa: BLE001
                for i in range(target_per_task):
                    samples.append(
                        {
                            "sample_id": f"{task_id}_fallback_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": "medium",
                            "prompt": _build_gsm8k_prompt(f"If you have {i+2} apples and buy 3 more, how many apples do you have?"),
                            "expected_answer": str(i + 5),
                            "grading_type": "numeric",
                            "source": "fallback-public-style",
                        }
                    )
            continue

        if task_id.startswith("mmlu"):
            try:
                if load_dataset is None:
                    raise RuntimeError("datasets package unavailable")
                ds = load_dataset(dataset_name, subset, split=split)
                for i, row in enumerate(ds.select(range(min(target_per_task, len(ds))))):
                    answer_idx = int(row["answer"])
                    label = ["A", "B", "C", "D"][answer_idx]
                    samples.append(
                        {
                            "sample_id": f"{task_id}_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": workload_class,
                            "prompt": _build_mmlu_prompt(row["question"], row["choices"]),
                            "expected_answer": label,
                            "grading_type": grading,
                            "source": dataset_name,
                        }
                    )
            except Exception:  # noqa: BLE001
                for i in range(target_per_task):
                    samples.append(
                        {
                            "sample_id": f"{task_id}_fallback_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": "short",
                            "prompt": _build_mmlu_prompt(
                                "Which letter comes first alphabetically?",
                                ["A", "B", "C", "D"],
                            ),
                            "expected_answer": "A",
                            "grading_type": "mcq",
                            "source": "fallback-public-style",
                        }
                    )
            continue

        if task_id.startswith("humaneval"):
            try:
                if load_dataset is None:
                    raise RuntimeError("datasets package unavailable")
                ds = load_dataset(dataset_name, subset, split=split)
                for i, row in enumerate(ds.select(range(min(target_per_task, len(ds))))):
                    expected = row.get("entry_point", "")
                    samples.append(
                        {
                            "sample_id": f"{task_id}_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": workload_class,
                            "prompt": _build_humaneval_prompt(row["prompt"]),
                            "expected_answer": expected,
                            "grading_type": "contains_diagnostic",
                            "source": dataset_name,
                        }
                    )
            except Exception:  # noqa: BLE001
                for i in range(target_per_task):
                    samples.append(
                        {
                            "sample_id": f"{task_id}_fallback_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": "medium",
                            "prompt": _build_humaneval_prompt("def add(a, b):\n    \"\"\"Return sum of two numbers.\"\"\"\n"),
                            "expected_answer": "add",
                            "grading_type": "contains_diagnostic",
                            "source": "fallback-public-style",
                        }
                    )
            continue

        if task_id.startswith("competition_math"):
            try:
                if load_dataset is None:
                    raise RuntimeError("datasets package unavailable")
                ds = load_dataset(dataset_name, subset, split=split)
                for i, row in enumerate(ds.select(range(min(target_per_task, len(ds))))):
                    problem = row.get("problem", "")
                    solution = row.get("solution", "")
                    samples.append(
                        {
                            "sample_id": f"{task_id}_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": workload_class,
                            "prompt": _build_gsm8k_prompt(problem),
                            "expected_answer": _extract_math_boxed(solution),
                            "grading_type": grading,
                            "source": dataset_name,
                        }
                    )
            except Exception:  # noqa: BLE001
                for i in range(target_per_task):
                    samples.append(
                        {
                            "sample_id": f"{task_id}_fallback_{i}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": "medium",
                            "prompt": _build_gsm8k_prompt(f"Compute {i+10} * 2."),
                            "expected_answer": str((i + 10) * 2),
                            "grading_type": "numeric",
                            "source": "fallback-public-style",
                        }
                    )
            continue

        # Keep long-context slice practical and deterministic.
        for i in range(target_per_task):
            samples.append(_make_long_context_public_style(i))

    samples = samples[:limit]
    output_path = run_dir / "raw" / "public_samples.jsonl"
    ensure_dir(output_path.parent)
    write_jsonl(output_path, samples)
    return samples
