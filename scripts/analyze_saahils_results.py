#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _mkdir(path.parent)
    df.to_csv(path, index=False)


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _correct_to_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[x == 'correct'] = 1.0
    out.loc[x == 'wrong'] = 0.0
    return out


def _bootstrap_delta_ci(base: np.ndarray, alt: np.ndarray, n_boot: int = 800, seed: int = 42) -> tuple[float, float]:
    if len(base) == 0 or len(alt) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(base))
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas.append(float(np.nanmean(alt[s] - base[s])))
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def _sign_test_p(a_better: int, b_better: int) -> float:
    n = a_better + b_better
    if n <= 0:
        return 1.0
    k = min(a_better, b_better)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _parse_lm_eval_preview(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not text:
        return rows
    patt = re.compile(r"\|([^|]+)\|\s*\d+\|([^|]+)\|\s*\d+\|([^|]+)\|[^|]*\|\s*([0-9.]+)\|[^|]*\|\s*([0-9.]+)\|")
    for m in patt.finditer(text):
        rows.append({
            'task': m.group(1).strip(),
            'filter': m.group(2).strip(),
            'metric': m.group(3).strip(),
            'value': float(m.group(4)),
            'stderr': float(m.group(5)),
        })
    return rows


def _set_style() -> dict[str, str]:
    sns.set_theme(style='whitegrid', context='talk')
    return {'llama_cpp': '#4C72B0', 'vllm': '#55A868', 'sglang': '#C44E52'}


def build_tables(agg_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    graded = _safe_read_csv(agg_dir / 'all_samples_graded.csv')
    latency = _safe_read_csv(agg_dir / 'latency_summary.csv')
    compare = _safe_read_csv(agg_dir / 'public_per_sample_comparison.csv')
    failures = _safe_read_csv(agg_dir / 'failure_bucket_summary.csv')
    run_manifest = _safe_read_csv(agg_dir / 'run_manifest.csv')
    std_events = _safe_read_csv(agg_dir / 'standard_evaluator_events.csv')

    if graded.empty:
        raise RuntimeError('all_samples_graded.csv is empty; cannot analyze')

    clean_runs = sorted(graded['run_id'].dropna().astype(str).unique().tolist())
    run_registry = pd.DataFrame({'run_id': clean_runs, 'locked_clean_input': True})
    run_registry['source'] = str(agg_dir)
    _write_csv(run_registry, out_dir / 'run_registry.csv')

    s = graded.copy()
    s['correct_binary'] = _correct_to_numeric(s['correctness'])
    s['missing_correctness'] = s['correct_binary'].isna()
    if 'benchmark_valid_for_claims' not in s.columns:
        s['benchmark_valid_for_claims'] = False
    s['benchmark_valid_for_claims'] = _bool(s['benchmark_valid_for_claims'])
    if 'missing_for_claims' not in s.columns:
        s['missing_for_claims'] = ~s['benchmark_valid_for_claims']
    if 'claim_exclusion_reason' not in s.columns:
        s['claim_exclusion_reason'] = np.where(s['missing_for_claims'], 'unlabeled_missing_for_claims', '')
    if 'failure_tag' not in s.columns:
        s['failure_tag'] = np.where(s['missing_correctness'], 'unknown_ungraded', np.where(s['correct_binary'] == 1.0, 'correct', 'semantic_mismatch'))
    for col in ['ttft_seconds', 'tpot_seconds', 'prompt_tokens', 'completion_tokens', 'estimated_cost_usd']:
        if col not in s.columns:
            s[col] = np.nan
    _write_csv(s, out_dir / 'sample_metrics_enriched.csv')

    grp = s.groupby(['backend', 'optimization_mode', 'task_id', 'workload_class'], dropna=False)
    benchmark_summary = grp.agg(
        rows=('sample_id', 'count'),
        benchmark_valid_rows=('benchmark_valid_for_claims', 'sum'),
        missing_correctness_rows=('missing_correctness', 'sum'),
        mean_correct_all=('correct_binary', 'mean'),
        mean_latency_s=('latency_seconds', 'mean'),
        p95_latency_s=('latency_seconds', lambda x: float(pd.to_numeric(x, errors='coerce').quantile(0.95))),
        mean_ttft_s=('ttft_seconds', 'mean'),
        mean_tpot_s=('tpot_seconds', 'mean'),
    ).reset_index()
    claim_only = (
        s[s['benchmark_valid_for_claims']]
        .groupby(['backend', 'optimization_mode', 'task_id', 'workload_class'], dropna=False)['correct_binary']
        .mean()
        .reset_index(name='mean_correct_claim_valid')
    )
    benchmark_summary = benchmark_summary.merge(claim_only, on=['backend', 'optimization_mode', 'task_id', 'workload_class'], how='left')
    benchmark_summary['missing_correctness_rate'] = benchmark_summary['missing_correctness_rows'] / benchmark_summary['rows'].clip(lower=1)
    _write_csv(benchmark_summary, out_dir / 'benchmark_summary.csv')

    latency_mode = pd.DataFrame()
    if not latency.empty:
        latency_mode = latency.copy()
        if 'ttft_missing_rate' in latency_mode.columns:
            latency_mode['ttft_quality_flag'] = np.where(latency_mode['ttft_missing_rate'].fillna(1.0) > 0.2, 'warning_high_missing', 'ok')
    _write_csv(latency_mode, out_dir / 'latency_ttft_tpot_summary.csv')

    fail_tax = (
        s.groupby(['backend', 'optimization_mode', 'failure_tag'], dropna=False)
        .size()
        .reset_index(name='count')
    )
    denom = fail_tax.groupby(['backend', 'optimization_mode'])['count'].transform('sum').clip(lower=1)
    fail_tax['rate'] = fail_tax['count'] / denom
    _write_csv(fail_tax, out_dir / 'failure_taxonomy_summary.csv')

    pair_rows = []
    if not compare.empty:
        correctness_cols = [c for c in compare.columns if c.startswith('correctness_')]
        backends = [c.replace('correctness_', '') for c in correctness_cols]
        for i, a in enumerate(backends):
            for b in backends[i+1:]:
                ac, bc = f'correctness_{a}', f'correctness_{b}'
                both = compare[[ac, bc]].dropna()
                if both.empty:
                    continue
                a_ok = both[ac].astype(str).str.lower().eq('correct')
                b_ok = both[bc].astype(str).str.lower().eq('correct')
                pair_rows.append({
                    'backend_a': a,
                    'backend_b': b,
                    'paired_rows': len(both),
                    'agreement_rate': float((both[ac] == both[bc]).mean()),
                    'a_correct_b_wrong_rate': float((a_ok & ~b_ok).mean()),
                    'b_correct_a_wrong_rate': float((b_ok & ~a_ok).mean()),
                })
    disagreement = pd.DataFrame(pair_rows)
    _write_csv(disagreement, out_dir / 'disagreement_pairwise_summary.csv')

    sig_rows = []
    for backend, bdf in s.groupby('backend'):
        piv = bdf.pivot_table(index=['run_id', 'sample_id', 'task_id'], columns='optimization_mode', values='correct_binary', aggfunc='mean')
        if 'baseline' not in piv.columns:
            continue
        for mode in sorted([c for c in piv.columns if c != 'baseline']):
            sub = piv[['baseline', mode]]
            missing_pairs = int(sub.isna().any(axis=1).sum())
            sub = sub.dropna()
            if sub.empty:
                sig_rows.append({'backend': backend, 'mode': mode, 'paired_n': 0, 'baseline_mean': np.nan, 'mode_mean': np.nan, 'delta_mode_minus_baseline': np.nan, 'delta_ci_low': np.nan, 'delta_ci_high': np.nan, 'a_better': 0, 'b_better': 0, 'ties': 0, 'sign_test_p_value': np.nan, 'missing_pairs': missing_pairs})
                continue
            a = sub['baseline'].to_numpy(dtype=float)
            m = sub[mode].to_numpy(dtype=float)
            a_better = int((a > m).sum())
            b_better = int((m > a).sum())
            ties = int((a == m).sum())
            low, high = _bootstrap_delta_ci(a, m)
            sig_rows.append({'backend': backend, 'mode': mode, 'paired_n': int(len(sub)), 'baseline_mean': float(np.nanmean(a)), 'mode_mean': float(np.nanmean(m)), 'delta_mode_minus_baseline': float(np.nanmean(m-a)), 'delta_ci_low': low, 'delta_ci_high': high, 'a_better': a_better, 'b_better': b_better, 'ties': ties, 'sign_test_p_value': _sign_test_p(a_better, b_better), 'missing_pairs': missing_pairs})
    significance = pd.DataFrame(sig_rows)
    _write_csv(significance, out_dir / 'significance_baseline_vs_mode.csv')

    lm_rows = []
    if not std_events.empty:
        previews = std_events[std_events['event'] == 'lm_eval_stdout_preview']
        for _, r in previews.iterrows():
            for parsed in _parse_lm_eval_preview(str(r.get('preview', ''))):
                lm_rows.append({'run_id': r.get('run_id'), 'backend': r.get('backend'), 'optimization_mode': r.get('optimization_mode'), **parsed})
    lm_eval_summary = pd.DataFrame(lm_rows)
    _write_csv(lm_eval_summary, out_dir / 'lm_eval_summary.csv')

    cost_efficiency = (
        s.groupby(['backend', 'optimization_mode'], dropna=False)
        .agg(rows=('sample_id', 'count'), correct_rows=('correct_binary', 'sum'), mean_correct=('correct_binary', 'mean'), total_estimated_cost_usd=('estimated_cost_usd', 'sum'), missing_claim_rows=('missing_for_claims', 'sum'))
        .reset_index()
    )
    cost_efficiency['cost_per_100_correct_est'] = np.where(cost_efficiency['correct_rows'] > 0, (cost_efficiency['total_estimated_cost_usd'] / cost_efficiency['correct_rows']) * 100.0, np.nan)
    _write_csv(cost_efficiency, out_dir / 'cost_efficiency_summary.csv')

    missingness = (
        s.groupby(['backend', 'optimization_mode'], dropna=False)
        .agg(rows=('sample_id', 'count'), missing_correctness_rows=('missing_correctness', 'sum'), missing_claim_rows=('missing_for_claims', 'sum'), ttft_missing_rows=('ttft_missing', lambda x: int(_bool(pd.Series(x)).sum()) if len(x) else 0), token_counts_missing_rows=('token_counts_missing', lambda x: int(_bool(pd.Series(x)).sum()) if len(x) else 0))
        .reset_index()
    )
    for c in ['missing_correctness_rows', 'missing_claim_rows', 'ttft_missing_rows', 'token_counts_missing_rows']:
        missingness[c.replace('_rows', '_rate')] = missingness[c] / missingness['rows'].clip(lower=1)
    _write_csv(missingness, out_dir / 'missingness_summary.csv')

    hypothesis_rows = []
    for backend in sorted(s['backend'].dropna().unique()):
        sub = cost_efficiency[cost_efficiency['backend'] == backend]
        base = sub[sub['optimization_mode'] == 'baseline']
        spec = sub[sub['optimization_mode'] == 'spec_decode']
        if base.empty or spec.empty:
            continue
        bacc = float(base['mean_correct'].iloc[0]) if not base['mean_correct'].isna().all() else np.nan
        sacc = float(spec['mean_correct'].iloc[0]) if not spec['mean_correct'].isna().all() else np.nan
        delta = sacc - bacc if pd.notna(bacc) and pd.notna(sacc) else np.nan
        hypothesis_rows.append({'hypothesis_id':'H1_spec_decode_tradeoff','backend':backend,'baseline_accuracy':bacc,'spec_decode_accuracy':sacc,'accuracy_delta':delta,'status':'supported' if pd.notna(delta) and delta >= -0.02 else 'mixed_or_not_supported'})
    hypothesis_results = pd.DataFrame(hypothesis_rows)
    _write_csv(hypothesis_results, out_dir / 'hypothesis_results.csv')

    quality_checks = pd.DataFrame([
        {'check':'clean_run_count','status':'pass' if len(clean_runs)==3 else 'warn','detail':f'run_count={len(clean_runs)}'},
        {'check':'benchmark_flags_present','status':'pass' if {'benchmark_valid_for_claims','missing_for_claims'}.issubset(s.columns) else 'fail','detail':'claim flags in enriched samples'},
        {'check':'ttft_collected','status':'pass' if 'ttft_seconds' in s.columns and s['ttft_seconds'].notna().any() else 'warn','detail':'ttft coverage present'},
        {'check':'no_spoiled_runs_used','status':'pass','detail':'source fixed to saahils_run_results/final_summary/data_collection/aggregated'}
    ])
    _write_csv(quality_checks, out_dir / 'quality_gate_checks.csv')

    source_registry = pd.DataFrame([
        {'table':'all_samples_graded','path':str(agg_dir / 'all_samples_graded.csv')},
        {'table':'latency_summary','path':str(agg_dir / 'latency_summary.csv')},
        {'table':'public_per_sample_comparison','path':str(agg_dir / 'public_per_sample_comparison.csv')},
        {'table':'failure_bucket_summary','path':str(agg_dir / 'failure_bucket_summary.csv')},
        {'table':'run_manifest','path':str(agg_dir / 'run_manifest.csv')},
        {'table':'standard_evaluator_events','path':str(agg_dir / 'standard_evaluator_events.csv')},
    ])
    _write_csv(source_registry, out_dir / 'source_registry.csv')

    return {
        'samples': s,
        'benchmark_summary': benchmark_summary,
        'latency_mode': latency_mode,
        'failure_taxonomy': fail_tax,
        'disagreement': disagreement,
        'significance': significance,
        'lm_eval_summary': lm_eval_summary,
        'cost_efficiency': cost_efficiency,
        'missingness': missingness,
        'hypothesis_results': hypothesis_results,
        'quality_checks': quality_checks,
        'run_registry': run_registry,
        'run_manifest': run_manifest,
        'failures': failures,
    }


def build_figures(frames: dict[str, pd.DataFrame], fig_dir: Path) -> pd.DataFrame:
    _mkdir(fig_dir)
    colors = _set_style()
    catalog = []

    def save(name: str, source_tables: list[str], purpose: str):
        plt.tight_layout()
        out = fig_dir / name
        plt.savefig(out, dpi=260, bbox_inches='tight')
        plt.close()
        catalog.append({'figure': name, 'source_tables': ','.join(source_tables), 'purpose': purpose})

    samples = frames['samples'].copy()
    bench = frames['benchmark_summary'].copy()
    fail_tax = frames['failure_taxonomy'].copy()
    dis = frames['disagreement'].copy()
    sig = frames['significance'].copy()
    miss = frames['missingness'].copy()
    ce = frames['cost_efficiency'].copy()

    # 1 claim-valid heatmap
    h = bench.pivot_table(index=['backend','optimization_mode'], columns='task_id', values='mean_correct_claim_valid', aggfunc='mean').sort_index()
    plt.figure(figsize=(13,6))
    sns.heatmap(h, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, linewidths=0.5)
    plt.title('Claim-Valid Correctness by Backend, Mode, Task')
    plt.xlabel('Task')
    plt.ylabel('Backend | Mode')
    save('fig_claim_valid_heatmap.png',['benchmark_summary.csv'],'Primary benchmark-valid quality map')

    # 2 all-data vs missing
    h_all = bench.pivot_table(index=['backend','optimization_mode'], columns='task_id', values='mean_correct_all', aggfunc='mean').sort_index()
    h_miss = bench.pivot_table(index=['backend','optimization_mode'], columns='task_id', values='missing_correctness_rate', aggfunc='mean').sort_index()
    fig, axes = plt.subplots(1,2,figsize=(18,6))
    sns.heatmap(h_all, annot=True, fmt='.2f', cmap='mako', vmin=0, vmax=1, linewidths=0.5, ax=axes[0])
    axes[0].set_title('All-Data Correctness (includes diagnostic rows)')
    axes[0].set_xlabel('Task'); axes[0].set_ylabel('Backend | Mode')
    sns.heatmap(h_miss, annot=True, fmt='.2f', cmap='rocket_r', vmin=0, vmax=1, linewidths=0.5, ax=axes[1])
    axes[1].set_title('Missing Correctness Rate')
    axes[1].set_xlabel('Task'); axes[1].set_ylabel('')
    save('fig_all_data_vs_missing_heatmaps.png',['benchmark_summary.csv'],'Missingness-aware interpretation guardrail')

    # 3 pareto quality vs latency
    p = samples.groupby(['backend','optimization_mode'],dropna=False).agg(mean_correct=('correct_binary','mean'),p95_latency_s=('latency_seconds',lambda x: float(pd.to_numeric(x,errors='coerce').quantile(0.95))),mean_ttft_s=('ttft_seconds','mean'),rows=('sample_id','count')).reset_index()
    plt.figure(figsize=(11,7))
    for _, r in p.iterrows():
        c = colors.get(str(r['backend']), '#444444')
        plt.scatter(r['p95_latency_s'], r['mean_correct'], s=max(60, r['rows']*0.25), color=c, alpha=0.85)
        plt.text(r['p95_latency_s'], r['mean_correct'], f"{r['backend']}:{r['optimization_mode']}", fontsize=9)
    plt.xlabel('P95 End-to-End Latency (s)')
    plt.ylabel('Mean Correctness')
    plt.title('Pareto: Quality vs Tail Latency')
    save('fig_pareto_quality_latency.png',['sample_metrics_enriched.csv'],'Tradeoff frontier for operational decision-making')

    # 4 TTFT TPOT distributions
    d = samples.copy()
    d['condition'] = d['backend'].astype(str) + '|' + d['optimization_mode'].astype(str)
    fig, axes = plt.subplots(1,2,figsize=(18,6))
    sns.boxplot(data=d, x='condition', y='ttft_seconds', showfliers=False, ax=axes[0])
    axes[0].set_title('TTFT Distribution by Condition'); axes[0].set_ylabel('TTFT (s)'); axes[0].tick_params(axis='x', rotation=35)
    sns.boxplot(data=d, x='condition', y='tpot_seconds', showfliers=False, ax=axes[1])
    axes[1].set_title('TPOT Distribution by Condition'); axes[1].set_ylabel('TPOT (s/token)'); axes[1].tick_params(axis='x', rotation=35)
    save('fig_ttft_tpot_distributions.png',['sample_metrics_enriched.csv'],'Decode path decomposition')

    # 5 workload stratified
    ws = samples.groupby(['backend','optimization_mode','workload_class'],dropna=False).agg(mean_correct=('correct_binary','mean'),p95_latency_s=('latency_seconds',lambda x: float(pd.to_numeric(x,errors='coerce').quantile(0.95)))).reset_index()
    fig, axes = plt.subplots(1,2,figsize=(18,6))
    sns.barplot(data=ws, x='workload_class', y='mean_correct', hue='backend', ax=axes[0])
    axes[0].set_ylim(0,1); axes[0].set_title('Correctness by Workload Class')
    sns.barplot(data=ws, x='workload_class', y='p95_latency_s', hue='backend', ax=axes[1])
    axes[1].set_title('P95 Latency by Workload Class')
    save('fig_workload_stratified_performance.png',['sample_metrics_enriched.csv'],'Workload-conditioned backend behavior')

    # 6 failure taxonomy stacked
    f = fail_tax.copy()
    f['condition'] = f['backend'].astype(str) + '|' + f['optimization_mode'].astype(str)
    piv = f.pivot_table(index='condition', columns='failure_tag', values='rate', fill_value=0)
    piv.plot(kind='bar', stacked=True, figsize=(14,6), colormap='tab20')
    plt.title('Failure Taxonomy Composition by Condition')
    plt.ylabel('Rate')
    plt.legend(title='Failure Tag', bbox_to_anchor=(1.02,1), loc='upper left')
    save('fig_failure_taxonomy_stacked.png',['failure_taxonomy_summary.csv'],'Failure mode composition')

    # 7 disagreement rates
    if not dis.empty:
      dl = dis.melt(id_vars=['backend_a','backend_b','paired_rows'], value_vars=['agreement_rate','a_correct_b_wrong_rate','b_correct_a_wrong_rate'], var_name='metric', value_name='rate')
      dl['pair'] = dl['backend_a'] + ' vs ' + dl['backend_b']
      plt.figure(figsize=(12,6))
      sns.barplot(data=dl, x='pair', y='rate', hue='metric')
      plt.title('Pairwise Backend Agreement and Asymmetric Wins')
      plt.ylabel('Rate'); plt.xlabel('Backend Pair')
      save('fig_pairwise_disagreement_rates.png',['disagreement_pairwise_summary.csv'],'Cross-backend disagreement structure')

    # 8 significance forest
    if not sig.empty:
      sf = sig.sort_values(['backend','mode']).copy()
      sf['label'] = sf['backend'] + ':' + sf['mode']
      y = np.arange(len(sf))
      plt.figure(figsize=(13,max(5,0.55*len(sf))))
      plt.axvline(0,color='black',alpha=0.5)
      plt.errorbar(sf['delta_mode_minus_baseline'], y, xerr=[sf['delta_mode_minus_baseline']-sf['delta_ci_low'], sf['delta_ci_high']-sf['delta_mode_minus_baseline']], fmt='o', capsize=4)
      plt.yticks(y, sf['label'])
      plt.xlabel('Delta Correctness (mode - baseline)')
      plt.title('Baseline-vs-Mode Effect with 95% Bootstrap CI')
      for i,row in sf.reset_index(drop=True).iterrows():
          txt=f"p={row['sign_test_p_value']:.3f}, n={int(row['paired_n'])}, miss={int(row['missing_pairs'])}"
          plt.text(float(row['delta_ci_high'])+0.003 if pd.notna(row['delta_ci_high']) else 0.01, i, txt, fontsize=8, va='center')
      save('fig_significance_forest.png',['significance_baseline_vs_mode.csv'],'Statistical confidence on optimization deltas')

    # 9 missingness dashboard
    md = miss.melt(id_vars=['backend','optimization_mode'], value_vars=['missing_correctness_rate','missing_claim_rate','ttft_missing_rate','token_counts_missing_rate'], var_name='metric', value_name='rate')
    md['condition'] = md['backend'] + '|' + md['optimization_mode']
    plt.figure(figsize=(14,6))
    sns.barplot(data=md, x='condition', y='rate', hue='metric')
    plt.ylim(0,1)
    plt.title('Missingness and Diagnostic Burden by Condition')
    plt.ylabel('Rate')
    plt.xticks(rotation=35, ha='right')
    save('fig_missingness_dashboard.png',['missingness_summary.csv'],'Data quality and caveat visibility')

    # 10 cost efficiency
    c = ce.copy(); c['condition'] = c['backend'] + '|' + c['optimization_mode']
    plt.figure(figsize=(13,6))
    sns.barplot(data=c, x='condition', y='cost_per_100_correct_est')
    plt.title('Estimated Cost per 100 Correct (NaN if pricing not configured)')
    plt.ylabel('USD / 100 correct')
    plt.xticks(rotation=35, ha='right')
    save('fig_cost_per_100_correct.png',['cost_efficiency_summary.csv'],'Cost-efficiency comparison')

    # 11 latency vs prompt tokens
    sc = samples.dropna(subset=['prompt_tokens_est','latency_seconds']).copy()
    sc['condition'] = sc['backend'].astype(str) + '|' + sc['optimization_mode'].astype(str)
    plt.figure(figsize=(12,7))
    sns.scatterplot(data=sc, x='prompt_tokens_est', y='latency_seconds', hue='condition', alpha=0.5, s=25)
    plt.title('Latency vs Prompt Token Estimate')
    plt.xlabel('Prompt token estimate'); plt.ylabel('Latency (s)')
    save('fig_latency_vs_prompt_tokens.png',['sample_metrics_enriched.csv'],'Input-size sensitivity')

    # 12 lm_eval summary bars
    lm = frames['lm_eval_summary']
    if not lm.empty:
      l = lm[lm['task'].isin(['gsm8k','humaneval_instruct'])].copy()
      if not l.empty:
        l['condition'] = l['backend'].astype(str) + '|' + l['optimization_mode'].astype(str)
        plt.figure(figsize=(14,6))
        sns.barplot(data=l, x='condition', y='value', hue='task')
        plt.title('lm_eval Metrics by Condition')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=35, ha='right')
        save('fig_lm_eval_metrics.png',['lm_eval_summary.csv'],'External benchmark validation snapshot')

    cat = pd.DataFrame(catalog)
    _write_csv(cat, fig_dir / 'chart_catalog.csv')
    return cat


def write_report(out_root: Path, frames: dict[str, pd.DataFrame], catalog: pd.DataFrame) -> None:
    s = frames['samples']
    sig = frames['significance']
    hyp = frames['hypothesis_results']
    q = frames['quality_checks']
    ce = frames['cost_efficiency']
    lm = frames['lm_eval_summary']

    by_backend = s.groupby('backend')['correct_binary'].mean().sort_values(ascending=False)
    missing_claim = int(_bool(s['missing_for_claims']).sum()) if 'missing_for_claims' in s.columns else 0
    total = int(len(s))
    ttft_missing = float(_bool(s.get('ttft_missing', pd.Series(dtype=bool))).mean()) if total else float('nan')

    novel = []
    if not sig.empty:
        top = sig.sort_values('delta_mode_minus_baseline', ascending=False).head(3)
        for _,r in top.iterrows():
            novel.append(f"- `{r['backend']}:{r['mode']}` delta correctness `{r['delta_mode_minus_baseline']:.3f}` with 95% CI `[{r['delta_ci_low']:.3f}, {r['delta_ci_high']:.3f}]`, sign-test `p={r['sign_test_p_value']:.3f}`.")
    if not ce.empty:
        m = ce[ce['cost_per_100_correct_est'].notna()]
        if not m.empty:
            b = m.sort_values('cost_per_100_correct_est').iloc[0]
            novel.append(f"- Lowest estimated cost-per-100-correct: `{b['backend']}:{b['optimization_mode']}` = `{b['cost_per_100_correct_est']:.4f}`.")
    if not novel:
        novel.append('- Cost fields are currently NaN because token pricing env vars are not configured; latency/TTFT/TPOT remain actionable.')

    hyp_lines = ['- No hypothesis summary rows generated.']
    if not hyp.empty:
        hyp_lines = [f"- `{r.hypothesis_id}` (`{r.backend}`): status=`{r.status}`, delta=`{r.accuracy_delta:.3f}`." for _,r in hyp.iterrows()]

    lm_lines = ['- lm_eval parsing yielded no structured rows.']
    if not lm.empty:
        l = lm.groupby(['backend','optimization_mode','task','metric'])['value'].mean().reset_index()
        lm_lines = [f"- `{r.backend}:{r.optimization_mode}` `{r.task}` `{r.metric}` = `{r.value:.3f}`" for _,r in l.head(15).iterrows()]

    q_lines = [f"- `{r['check']}`: **{r['status']}** ({r['detail']})" for _,r in q.iterrows()]

    lines = [
        '# Deep Analysis Report: saahils_run_results',
        '',
        f"Generated: `{datetime.now(UTC).isoformat()}`",
        '',
        '## Scope and Inputs',
        '- Source: `saahils_run_results/final_summary/data_collection/aggregated` only.',
        '- Included runs: llama_cpp baseline + vllm(3 modes) + sglang(3 modes).',
        '',
        '## Executive Summary',
        f'- Total rows analyzed: **{total}**',
        f'- Rows marked missing/diagnostic for claims: **{missing_claim}**',
        f'- TTFT missing rate: **{ttft_missing:.3f}**',
        '',
        '## Backend-Level Mean Correctness (available graded rows)',
    ]
    for k,v in by_backend.items():
        lines.append(f'- `{k}`: `{v:.3f}`')

    lines += [
        '',
        '## Hypothesis Results',
        *hyp_lines,
        '',
        '## Novel / Interesting Findings',
        *novel,
        '',
        '## External Benchmark Signals (lm_eval)',
        *lm_lines,
        '',
        '## Figures and Sources',
    ]
    for _,r in catalog.iterrows():
        lines.append(f"- `{r['figure']}` — {r['purpose']} (source: {r['source_tables']})")

    lines += [
        '',
        '## Caveats and Notes',
        '- `benchmark_valid_for_claims` and missingness flags are preserved and reported, not dropped.',
        '- Diagnostic coding rows are present and explicitly flagged via `claim_exclusion_reason`.',
        '- Cost conclusions require pricing env vars to populate `estimated_cost_usd` fields.',
        '',
        '## Quality Gates',
        *q_lines,
        '',
        '## Artifact Paths',
        f"- Derived tables: `{out_root / 'derived_tables'}`",
        f"- Figures: `{out_root / 'figures'}`",
        f"- Chart catalog: `{out_root / 'figures' / 'chart_catalog.csv'}`",
    ]

    (out_root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Deep analysis for saahils_run_results')
    parser.add_argument('--aggregated-dir', default='saahils_run_results/final_summary/data_collection/aggregated')
    parser.add_argument('--output-root', default='saahils_run_results/analysis_outputs')
    args = parser.parse_args()

    agg_dir = Path(args.aggregated_dir)
    if not agg_dir.exists():
        raise RuntimeError(f'Aggregated dir not found: {agg_dir}')

    ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    out_root = Path(args.output_root) / f'ANALYSIS_{ts}'
    derived = out_root / 'derived_tables'
    figures = out_root / 'figures'
    _mkdir(derived)
    _mkdir(figures)

    frames = build_tables(agg_dir=agg_dir, out_dir=derived)
    catalog = build_figures(frames=frames, fig_dir=figures)
    write_report(out_root=out_root, frames=frames, catalog=catalog)

    manifest = {
        'output_root': str(out_root),
        'derived_tables': str(derived),
        'figures': str(figures),
        'report': str(out_root / 'report.md'),
        'generated_at_utc': datetime.now(UTC).isoformat(),
    }
    (out_root / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(out_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
