#!/usr/bin/env python3
"""Exploratory template-cluster sensitivity from frozen per-question outputs.

No model calls, hypothesis tests, or changes to existing multiplicity families.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from analyze_template_clusters import question_template, index_rows, cluster_bootstrap
from analyze_revision import normalize, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    inputs = []

    def read(name):
        path = ROOT / 'results' / name
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload['status'] != 'PASS':
            raise ValueError(f'Non-PASS input: {name}')
        inputs.append({'path': 'results/' + name, 'sha256': hashlib.sha256(raw).hexdigest()})
        return payload['results']

    primary = {b: index_rows(read(f'qwen3_awq_test300_B{b}.json'), 'ra_mmr')
               for b in (128, 256, 512)}
    ablations = read('clean_component_ablation_qwen_B256.json')
    seed = 20261024
    iterations = 20000
    rng = np.random.default_rng(seed)
    comparisons = []

    def analyze(name, budget, left, right, endpoint=False):
        if set(left) != set(right) or len(left) != 900:
            raise ValueError(f'Pairing mismatch: {name}')
        if Counter(k[0] for k in left) != {1: 300, 2: 300, 3: 300}:
            raise ValueError(f'Hop mismatch: {name}')
        grouped = defaultdict(list)
        for key in sorted(left):
            a, b = left[key], right[key]
            if a['question'] != b['question']:
                raise ValueError(f'Question mismatch: {name} {key}')
            score = metrics([normalize(x) for x in a['selected_endpoints']],
                            [normalize(x) for x in a['gold_answers']])['f1'] if endpoint else float(a['f1'])
            grouped[(key[0], question_template(a['question']))].append(score - float(b['f1']))
        clusters = [np.asarray(v, dtype=float) for v in grouped.values()]
        lo, hi = cluster_bootstrap(clusters, iterations, rng)
        comparisons.append({
            'comparison': name, 'budget': budget, 'questions': len(left),
            'template_clusters': len(clusters), 'largest_cluster': max(map(len, clusters)),
            'question_weighted_difference': float(np.concatenate(clusters).mean()),
            'cluster_bootstrap_95_ci_low': lo, 'cluster_bootstrap_95_ci_high': hi,
        })

    for variant in ('racps_no_relation', 'racps_no_cost', 'racps_no_novelty'):
        analyze('full_minus_' + variant, 256, primary[256], index_rows(ablations, variant))
    for budget in (128, 256, 512):
        analyze('endpoint_minus_reader', budget, primary[budget], primary[budget], endpoint=True)
    result = {
        'status': 'PASS',
        'analysis': {
            'classification': 'exploratory sensitivity; pointwise intervals; no new significance tests',
            'template_definition': 'casefold + every bracketed entity -> [ENTITY] + whitespace collapse',
            'cluster_key': '(official hop, normalized question template)',
            'estimand': 'question-weighted paired macro answer-set F1 difference',
            'bootstrap': 'resample all 393 template clusters with replacement; concatenate member differences',
            'hop_weights': 'original sample has 300 questions per hop; resampled hop weights are not held fixed',
            'relationship_to_primary': 'matches the pooled cluster bootstrap in analyze_template_clusters.py; does not alter primary or component Holm families',
            'seed': seed, 'bootstrap_iterations': iterations,
        },
        'inputs': inputs, 'comparisons': comparisons,
    }
    target = ROOT / 'derived' / 'exploratory_cluster_sensitivity.json'
    if args.verify:
        if json.loads(target.read_text()) != result:
            raise ValueError('Recomputed exploratory intervals differ from the saved record')
        print('PASS: six exploratory cluster intervals independently recomputed from frozen inputs')
    else:
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        with target.with_suffix('.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
