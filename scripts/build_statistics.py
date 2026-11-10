#!/usr/bin/env python3
"""Compute analysis inputs from local runs, without publishing or downloading data."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    args = ap.parse_args()
    root = args.workspace.resolve()
    jobs = []
    for budget in (128, 256, 512):
        jobs.append((f'stats_qwen3_awq_test300_B{budget}', [
            'src/analyze_paired_results.py', '--input',
            f'results/qwen3_awq_test300_B{budget}.json', '--seed', '20260818',
            '--bootstrap-iterations', '10000', '--permutation-iterations', '50000']))
    for seed in (20260819,20260820):
        jobs.append((f'stats_qwen3_awq_test300_B256_seed{seed}', [
            'src/analyze_paired_results.py','--input',f'results/qwen3_awq_test300_B256_seed{seed}.json',
            '--seed',str(seed),'--bootstrap-iterations','10000','--permutation-iterations','50000']))
    jobs.append(('stats_phi4mini_test300_B256_seed20260818',[
        'src/analyze_paired_results.py','--input','results/phi4mini_test300_B256_seed20260818.json',
        '--seed','20260819','--bootstrap-iterations','10000','--permutation-iterations','50000']))
    jobs.extend([
        ('stats_clean_component_ablation_qwen_B256', [
            'src/analyze_component_ablation.py', '--primary-results',
            'results/qwen3_awq_test300_B256.json', '--ablation-results',
            'results/clean_component_ablation_qwen_B256.json', '--seed', '20260823',
            '--bootstrap-iterations', '10000', '--permutation-iterations', '50000']),
        ('template_cluster_stats_qwen_main', [
            'src/analyze_template_clusters.py', '--results',
            *[f'results/qwen3_awq_test300_B{b}.json' for b in (128, 256, 512)],
            '--seed', '20260823', '--bootstrap-iterations', '20000',
            '--permutation-iterations', '100000']),
        ('stats_webqsp_official_chain_qwen_test500_B256', [
            'src/analyze_webqsp_replication.py', '--input',
            'results/week2_webqsp_qwen_test500_B256.json', '--seed', '20260823',
            '--bootstrap-iterations', '10000', '--permutation-iterations', '50000']),
        ('cross_model_replication_B256_aligned_seed20260818',[
            'src/analyze_cross_model_replication.py','--qwen-results','results/qwen3_awq_test300_B256.json',
            '--qwen-stats','results/stats_qwen3_awq_test300_B256.json',
            '--phi-results','results/phi4mini_test300_B256_seed20260818.json',
            '--phi-stats','results/stats_phi4mini_test300_B256_seed20260818.json',
            '--seed','20260819','--bootstrap-iterations','10000','--permutation-iterations','50000']),
        ('stats_week2_shared_evidence_B256',[
            'src/analyze_shared_evidence.py','--qwen-reader-specific','results/qwen3_awq_test300_B256.json',
            '--phi-reader-specific','results/phi4mini_test300_B256_seed20260818.json',
            '--phi-shared-evidence','results/week2_shared_evidence_phi_qwenpaths_test300_B256.json',
            '--seed','20260818','--bootstrap-iterations','10000','--permutation-iterations','50000'])])
    generated={f'results/{name}.json' for name,_ in jobs}
    for _,command in jobs:
        for word in command:
            if word.endswith('.json') and word not in generated and not (root/word).is_file():
                raise SystemExit(f'Missing experiment input: {word}')
    for name, _ in jobs:
        for ext in ('json', 'csv'):
            if (root / 'results' / f'{name}.{ext}').exists():
                raise SystemExit(f'Refusing to replace existing analysis: {name}.{ext}')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    for name, command in jobs:
        with (root / 'logs' / f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, *command, '--output-json',
                            f'results/{name}.json', '--output-csv', f'results/{name}.csv'],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Computed {name}', flush=True)
    print('These are newly computed statistics, not independent frozen references.')

if __name__ == '__main__':
    main()
