#!/usr/bin/env python3
"""
Build an optimization regression report: baseline correct → optimized wrong (same engine),
plus cross-optimization patterns. Includes GSM8K problem text, expected answer, extracted
final numerics, and response tails per mode.

Writes:
  report/optimization_regression_analysis.md
  processed/optimization_regression_detail.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.normalize_outputs import extract_numeric


def _is_correct(v: object) -> bool:
    return str(v or "").strip().lower() == "correct"


def _is_wrong(v: object) -> bool:
    return str(v or "").strip().lower() == "wrong"


def _problem_excerpt(prompt: str, max_chars: int = 900) -> str:
    p = str(prompt or "")
    if "Problem:" in p:
        tail = p.split("Problem:", 1)[1]
        if "\n\nAnswer:" in tail:
            tail = tail.split("\n\nAnswer:", 1)[0]
        body = tail.strip()
    else:
        body = p.strip()
    return body[:max_chars] + ("…" if len(body) > max_chars else "")


def _response_tail(text: object, n: int = 550) -> str:
    s = str(text or "").replace("\r\n", "\n").strip()
    if len(s) <= n:
        return s
    return "…" + s[-n:]


def _numeric_cell(text: object) -> str:
    x = extract_numeric(str(text or ""))
    return x if x is not None else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/20260429T230230Z_black_box_vllm_sglang"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    graded_path = run_dir / "processed" / "all_samples_graded.csv"
    if not graded_path.exists():
        raise FileNotFoundError(graded_path)

    df = pd.read_csv(graded_path)
    df = df[df["task_id"].astype(str).str.startswith("gsm8k")].copy()
    if df.empty:
        raise RuntimeError("No GSM8K rows in graded table.")

    backends = ["vllm", "sglang"]
    modes = ["baseline", "kv_cache_quant", "spec_decode"]

    rows_out: list[dict[str, object]] = []
    md: list[str] = [
        "# Optimization-induced regressions (same engine)",
        "",
        "Definitions:",
        "",
        "- **Regression**: that engine’s **baseline** graded **correct**, while an **optimization mode** graded **wrong**.",
        "- **Cross-opt**: how KV-quant vs speculative decoding disagree when baseline was correct.",
        "- Answers shown use the same **last-number** GSM8K heuristic as grading (`extract_numeric`). Response snippets are **tails** (final ~550 chars).",
        "",
        "**What “baseline” means:** Under **`## Endpoint: vllm`**, **baseline** is the **vLLM** deployment with `optimization_mode=baseline` (**no** KV quantization, **no** speculative decoding)—not llama.cpp. Under **`## Endpoint: sglang`**, **baseline** is the same pattern for **SGLang**. Table columns labeled **`baseline`** / **`kv_cache_quant`** / **`spec_decode`** are those three deployments for that endpoint only.",
        "",
        "---",
        "",
    ]

    for backend in backends:
        sub = df[df["backend"].astype(str).str.lower() == backend].copy()
        if sub.empty:
            continue

        wide: dict[str, dict[str, dict[str, object]]] = {}
        for _, r in sub.iterrows():
            sid = str(r["sample_id"])
            mode = str(r["optimization_mode"])
            if mode not in modes:
                continue
            wide.setdefault(sid, {})[mode] = {
                "correctness": r["correctness"],
                "response_text": r["response_text"],
                "prompt": r["prompt"],
                "expected_answer": r["expected_answer"],
            }

        complete_ids = sorted(
            [sid for sid, d in wide.items() if all(m in d for m in modes)],
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
        )

        md.append(f"## Endpoint: **{backend}**")
        md.append("")
        md.append(f"_Samples with baseline + KV + spec rows: **{len(complete_ids)}**_")
        md.append("")

        kv_reg: list[str] = []
        spec_reg: list[str] = []
        kv_only: list[str] = []
        spec_only: list[str] = []
        both_reg: list[str] = []
        baseline_wrong_kv_ok: list[str] = []
        baseline_wrong_spec_ok: list[str] = []

        for sid in complete_ids:
            d = wide[sid]
            cb = d["baseline"]["correctness"]
            ck = d["kv_cache_quant"]["correctness"]
            cs = d["spec_decode"]["correctness"]

            if _is_correct(cb) and _is_wrong(ck):
                kv_reg.append(sid)
            if _is_correct(cb) and _is_wrong(cs):
                spec_reg.append(sid)
            if _is_correct(cb) and _is_wrong(ck) and _is_correct(cs):
                kv_only.append(sid)
            if _is_correct(cb) and _is_wrong(cs) and _is_correct(ck):
                spec_only.append(sid)
            if _is_correct(cb) and _is_wrong(ck) and _is_wrong(cs):
                both_reg.append(sid)
            if _is_wrong(cb) and _is_correct(ck):
                baseline_wrong_kv_ok.append(sid)
            if _is_wrong(cb) and _is_correct(cs):
                baseline_wrong_spec_ok.append(sid)

        md.append("### Summary counts")
        md.append("")
        md.append("| Pattern | Count |")
        md.append("|---------|-------|")
        md.append(f"| Baseline ✓ → **KV quant ✗** | {len(kv_reg)} |")
        md.append(f"| Baseline ✓ → **Spec decode ✗** | {len(spec_reg)} |")
        md.append(f"| Baseline ✓ → KV ✗ only (spec still ✓) | {len(kv_only)} |")
        md.append(f"| Baseline ✓ → Spec ✗ only (KV still ✓) | {len(spec_only)} |")
        md.append(f"| Baseline ✓ → **both** KV ✗ and Spec ✗ | {len(both_reg)} |")
        md.append(f"| Baseline ✗ → KV ✓ (optimization helped) | {len(baseline_wrong_kv_ok)} |")
        md.append(f"| Baseline ✗ → Spec ✓ (optimization helped) | {len(baseline_wrong_spec_ok)} |")
        md.append("")

        def emit_detail_section(title: str, ids: list[str], tag: str) -> None:
            if not ids:
                return
            md.append(f"### {title}")
            md.append("")
            for sid in ids:
                d = wide[sid]
                exp = str(d["baseline"]["expected_answer"])
                prob = _problem_excerpt(str(d["baseline"]["prompt"]))
                nb = _numeric_cell(d["baseline"]["response_text"])
                nk = _numeric_cell(d["kv_cache_quant"]["response_text"])
                ns = _numeric_cell(d["spec_decode"]["response_text"])
                md.append(f"#### `{sid}`")
                md.append("")
                md.append(f"- **Expected (graded):** `{exp}` (extracted `{_numeric_cell(exp)}`)")
                md.append(f"- **Extracted finals:** baseline `{nb}` · KV `{nk}` · spec `{ns}`")
                md.append(
                    f"- **Correctness:** baseline `{d['baseline']['correctness']}` · "
                    f"KV `{d['kv_cache_quant']['correctness']}` · spec `{d['spec_decode']['correctness']}`"
                )
                md.append("")
                md.append("**Problem**")
                md.append("")
                md.append(f"> {prob.replace(chr(10), ' ')}")
                md.append("")
                md.append("| Mode | Response tail |")
                md.append("|------|----------------|")
                for mode, label in [
                    ("baseline", "baseline"),
                    ("kv_cache_quant", "kv_cache_quant"),
                    ("spec_decode", "spec_decode"),
                ]:
                    tail = _response_tail(d[mode]["response_text"]).replace("|", "\\|").replace("\n", "<br>")
                    md.append(f"| {label} | {tail} |")
                md.append("")

                rows_out.append(
                    {
                        "endpoint_backend": backend,
                        "category": tag,
                        "sample_id": sid,
                        "expected_answer": exp,
                        "expected_numeric": _numeric_cell(exp),
                        "numeric_baseline": nb,
                        "numeric_kv_cache_quant": nk,
                        "numeric_spec_decode": ns,
                        "correctness_baseline": d["baseline"]["correctness"],
                        "correctness_kv_cache_quant": d["kv_cache_quant"]["correctness"],
                        "correctness_spec_decode": d["spec_decode"]["correctness"],
                        "problem_excerpt": prob.replace("\n", " ")[:2000],
                    }
                )

        emit_detail_section(
            "Baseline ✓ → **both** KV ✗ and Spec ✗",
            both_reg,
            "both_optimizations_regress",
        )
        emit_detail_section(
            "Baseline ✓ → KV ✗ only (spec still ✓)",
            kv_only,
            "kv_only_regression",
        )
        emit_detail_section(
            "Baseline ✓ → Spec ✗ only (KV still ✓)",
            spec_only,
            "spec_only_regression",
        )

        md.append("---")
        md.append("")

    md.append("## Cross-endpoint comparison (optimization regressions)")
    md.append("")
    md.append(
        "Prompts where **vLLM** shows a regression but **SGLang** does not at the same optimization (both baselines correct), and the converse."
    )
    md.append("")

    def load_wide_backend(backend: str) -> dict[str, dict[str, dict]]:
        sub = df[df["backend"].astype(str).str.lower() == backend]
        out: dict[str, dict[str, dict]] = {}
        for _, r in sub.iterrows():
            sid = str(r["sample_id"])
            mode = str(r["optimization_mode"])
            out.setdefault(sid, {})[mode] = {"correctness": r["correctness"]}
        return out

    wv = load_wide_backend("vllm")
    ws = load_wide_backend("sglang")

    def ids_kv_reg(w: dict[str, dict[str, dict]]) -> set[str]:
        sids: set[str] = set()
        for sid, d in w.items():
            if not all(m in d for m in modes):
                continue
            if _is_correct(d["baseline"]["correctness"]) and _is_wrong(d["kv_cache_quant"]["correctness"]):
                sids.add(sid)
        return sids

    def ids_spec_reg(w: dict[str, dict[str, dict]]) -> set[str]:
        sids: set[str] = set()
        for sid, d in w.items():
            if not all(m in d for m in modes):
                continue
            if _is_correct(d["baseline"]["correctness"]) and _is_wrong(d["spec_decode"]["correctness"]):
                sids.add(sid)
        return sids

    vk = ids_kv_reg(wv)
    sk = ids_kv_reg(ws)
    vs_set = ids_spec_reg(wv)
    ss_set = ids_spec_reg(ws)

    md.append("| Contrast | sample_ids |")
    md.append("|----------|------------|")
    md.append(f"| KV regression **vLLM only** (not SGLang) | `{sorted(vk - sk)}` |")
    md.append(f"| KV regression **SGLang only** (not vLLM) | `{sorted(sk - vk)}` |")
    md.append(f"| Spec regression **vLLM only** | `{sorted(vs_set - ss_set)}` |")
    md.append(f"| Spec regression **SGLang only** | `{sorted(ss_set - vs_set)}` |")
    md.append("")

    # llama.cpp (baseline-only): agreement with optimization regressions on GPU stacks
    md.append("## llama.cpp baseline vs GPU optimization regressions")
    md.append("")
    md.append(
        "Where **llama.cpp** (same-task baseline) is **correct**, but a GPU stack shows **KV** or **spec** regression. "
        "This is *not* proof of causation—only prompt-level alignment useful for triage."
    )
    md.append("")

    llama_rows = df[
        (df["backend"].astype(str).str.lower() == "llama_cpp") & (df["optimization_mode"].astype(str) == "baseline")
    ]
    lm_correct = {str(r["sample_id"]) for _, r in llama_rows.iterrows() if _is_correct(r["correctness"])}

    def regress_sets_for(backend_name: str) -> tuple[set[str], set[str]]:
        w = load_wide_backend(backend_name)
        kv_r: set[str] = set()
        sp_r: set[str] = set()
        for sid, d in w.items():
            if not all(m in d for m in modes):
                continue
            if _is_correct(d["baseline"]["correctness"]) and _is_wrong(d["kv_cache_quant"]["correctness"]):
                kv_r.add(sid)
            if _is_correct(d["baseline"]["correctness"]) and _is_wrong(d["spec_decode"]["correctness"]):
                sp_r.add(sid)
        return kv_r, sp_r

    for bn in ("vllm", "sglang"):
        kv_r, sp_r = regress_sets_for(bn)
        md.append(f"### llama ✓ and **{bn}** KV regression")
        md.append("")
        md.append(f"- `{sorted(sid for sid in kv_r if sid in lm_correct)}`")
        md.append("")
        md.append(f"### llama ✓ and **{bn}** spec regression")
        md.append("")
        md.append(f"- `{sorted(sid for sid in sp_r if sid in lm_correct)}`")
        md.append("")

    out_csv = run_dir / "processed" / "optimization_regression_detail.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    else:
        out_csv.write_text("", encoding="utf-8")

    report_path = run_dir / "report" / "optimization_regression_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {out_csv} ({len(rows_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
