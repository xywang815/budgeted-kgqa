#!/usr/bin/env python3
"""Export precise means by recorded condition, hop/chain, budget and decoding.

Example: --input qwen-seed20260818=/path/results.json --output metrics.csv
No weighting across runs; each labeled input remains a separate experiment.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


def export(inputs, output):
    records = []
    for item in inputs:
        label, name = item.split('=', 1)
        rows = json.loads(Path(name).read_text())['results']
        groups = defaultdict(list)
        for r in rows:
            for stratum in ['all', str(r.get('hop', r.get('annotated_max_hops', 'unknown')))]:
                key = (r['condition'], r.get('budget', 256), r.get('decoding', 'greedy'),
                       r.get('sampling_seed', ''), r.get('group', ''), stratum)
                groups[key].append(r)
        for key, rr in sorted(groups.items(), key=lambda x: str(x[0])):
            c, b, decoding, seed, sample, stratum = key
            row = dict(experiment=label, condition=c, budget=b, decoding=decoding,
                       decoding_seed=seed, sample_group=sample, stratum=stratum, n=len(rr))
            for metric in ['precision', 'recall', 'f1', 'evidence_tokens', 'total_input_tokens',
                           'output_tokens', 'selected_paths', 'generation_seconds']:
                row[metric] = mean(r[metric] for r in rr) if all(metric in r for r in rr) else ''
            for metric in ['precision', 'recall', 'f1']:
                values = [r.get('selected_endpoint_' + metric, r.get('endpoint_metrics', {}).get(metric)) for r in rr]
                row['endpoint_' + metric] = mean(values) if all(v is not None for v in values) else ''
            values = [r.get('method_retrieval_seconds', r.get('retrieval_seconds')) for r in rr]
            row['retrieval_seconds'] = mean(values) if all(v is not None for v in values) else ''
            row['hit_generation_cap'] = sum(r['output_hit_max_new_tokens'] for r in rr) if all('output_hit_max_new_tokens' in r for r in rr) else ''
            records.append(row)
    with output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print(len(records), 'exact aggregate rows written to', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', action='append', required=True, help='label=original-json-path')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); export(a.input, a.output)
