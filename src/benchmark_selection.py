from __future__ import annotations

import os
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


def _build_longbench_v2_prompt(
    context: str,
    question: str,
    choice_a: str,
    choice_b: str,
    choice_c: str,
    choice_d: str,
) -> str:
    return (
        "Read the long context below, then answer the multiple-choice question. "
        "Reply with only one letter: A, B, C, or D.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"A. {choice_a}\n"
        f"B. {choice_b}\n"
        f"C. {choice_c}\n"
        f"D. {choice_d}\n\n"
        "Answer:"
    )


def _longbench_v2_sample_indices(ds: Any, need: int, max_context_chars: int, seed: int, truncate_keep: int) -> list[int]:
    eligible: list[int] = []
    for i in range(len(ds)):
        ctx = str(ds[i]["context"])
        eff = _effective_longbench_context(ctx, truncate_keep)
        if len(eff) <= max_context_chars:
            eligible.append(i)
    if len(eligible) < need:
        raise ValueError(
            f"LongBench-v2: only {len(eligible)} rows fit after truncation (keep={truncate_keep}) "
            f"and max_context_chars={max_context_chars}; need {need}. "
            "Raise LONGBENCH_TRUNCATE_CONTEXT_CHARS or LONGBENCH_MAX_CONTEXT_CHARS, or lower LIMIT."
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:need]


def _effective_longbench_context(raw: str, truncate_keep: int) -> str:
    """Optionally shrink context so prompts fit smaller GPU context windows (e.g. 8k-token L4)."""
    s = str(raw)
    if truncate_keep > 0 and len(s) > truncate_keep:
        return (
            "[Earlier context omitted — last segment only, for hardware limits.]\n\n" + s[-truncate_keep:]
        )
    return s


def _make_long_context_public_style(task_id: str, idx: int) -> dict[str, Any]:
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
        "sample_id": f"{task_id}_{idx}",
        "workload": "public",
        "task_id": task_id,
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

        if task_id.startswith("longbench"):
            try:
                if load_dataset is None:
                    raise RuntimeError("datasets package unavailable")
                ds = load_dataset(dataset_name, subset, split=split)
                max_chars = int(os.getenv("LONGBENCH_MAX_CONTEXT_CHARS", "120000"))
                truncate_keep = int(os.getenv("LONGBENCH_TRUNCATE_CONTEXT_CHARS", "0"))
                seed = int(os.getenv("LONGBENCH_SELECTION_SEED", "42"))
                take = min(target_per_task, len(ds))
                pick = _longbench_v2_sample_indices(ds, take, max_chars, seed, truncate_keep)
                for i in pick:
                    row = ds[i]
                    safe_id = str(row["_id"]).replace("/", "_").replace("\\", "_")
                    ans = str(row["answer"]).strip().upper()
                    if ans not in {"A", "B", "C", "D"}:
                        ans = ans[:1].upper()
                    ctx_used = _effective_longbench_context(str(row["context"]), truncate_keep)
                    samples.append(
                        {
                            "sample_id": f"{task_id}_{safe_id}",
                            "workload": "public",
                            "task_id": task_id,
                            "workload_class": workload_class,
                            "prompt": _build_longbench_v2_prompt(
                                ctx_used,
                                str(row["question"]),
                                str(row["choice_A"]),
                                str(row["choice_B"]),
                                str(row["choice_C"]),
                                str(row["choice_D"]),
                            ),
                            "expected_answer": ans[:1],
                            "grading_type": grading,
                            "source": dataset_name,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"LongBench task {task_id!r}: failed to load dataset {dataset_name!r} "
                    "(check HF access and datasets package)." 
                ) from exc
            continue

        # Fallback for unknown benchmark ids: deterministic synthetic needle.
        for fallback_i in range(target_per_task):
            samples.append(_make_long_context_public_style(task_id, fallback_i))

    samples = samples[:limit]
    output_path = run_dir / "raw" / "public_samples.jsonl"
    ensure_dir(output_path.parent)
    write_jsonl(output_path, samples)
    return samples
