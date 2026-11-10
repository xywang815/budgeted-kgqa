#!/usr/bin/env python3
"""Compare freshly rebuilt analyses with frozen statistics, including input hashes."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
PAIRS = {
    'primary_B128.json': 'stats_qwen3_awq_test300_B128.json',
    'primary_B256.json': 'stats_qwen3_awq_test300_B256.json',
    'primary_B512.json': 'stats_qwen3_awq_test300_B512.json',
    'clean_components.json': 'stats_clean_component_ablation_qwen_B256.json',
    'template_clusters.json': 'template_cluster_stats_qwen_main.json',
    'legacy_webqsp.json': 'stats_webqsp_official_chain_qwen_test500_B256.json',
}


def semantic_payload(value):
    """Exclude only known run timestamps and input locations; keep input digests."""
    value = copy.deepcopy(value)
    value.pop('created_utc', None)
    value.pop('input', None)
    inputs = value.get('inputs')
    if isinstance(inputs, dict):
        for name in ('primary', 'ablation'):
            inputs.pop(name, None)
    elif isinstance(inputs, list):
        for item in inputs:
            item.pop('path', None)
    return value


def differences(left, right, prefix='$'):
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(left.keys() | right.keys()):
            path = prefix + '.' + key
            if key not in left or key not in right:
                out.append(path + ' (missing key)')
            else:
                out.extend(differences(left[key], right[key], path))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix + ' (different length)']
        return [d for i, (a, b) in enumerate(zip(left, right))
                for d in differences(a, b, f'{prefix}[{i}]')]
    # Exact equality preserves the published rounding and Monte Carlo outputs.
    return [] if type(left) is type(right) and left == right else [prefix]


def verify(rebuilt_dir=None, results_dir=None):
    rebuilt_dir = Path(rebuilt_dir or ROOT / 'rebuilt')
    results_dir = Path(results_dir or ROOT / 'results')
    checks = []
    for fresh_name, frozen_name in PAIRS.items():
        fresh, frozen = rebuilt_dir / fresh_name, results_dir / frozen_name
        row = {'rebuilt': fresh_name, 'frozen': frozen_name}
        if not fresh.is_file() or not frozen.is_file():
            row.update(status='FAIL', differences=['required file missing'])
        else:
            try:
                a, b = json.loads(fresh.read_text()), json.loads(frozen.read_text())
                diff = differences(semantic_payload(a), semantic_payload(b))
                if a.get('status') != 'PASS' or b.get('status') != 'PASS':
                    diff.append('$.status (analysis not PASS)')
                row.update(status='FAIL' if diff else 'PASS', differences=diff,
                           rebuilt_sha256=hashlib.sha256(fresh.read_bytes()).hexdigest(),
                           frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
            except (ValueError, TypeError) as exc:
                row.update(status='FAIL', differences=[str(exc)])
        checks.append(row)
    return {'status': 'PASS' if all(r['status'] == 'PASS' for r in checks) else 'FAIL',
            'comparison': 'exact JSON equality except documented provenance locations and created_utc',
            'input_hashes_compared': True, 'checks': checks}


def require_if_rebuilt():
    """Reject incomplete or mismatched rebuilds before downstream regeneration."""
    if any((ROOT / 'rebuilt' / name).exists() for name in PAIRS):
        result = verify()
        if result['status'] != 'PASS':
            raise RuntimeError('Fresh statistical verification failed: ' + json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuilt-dir', type=Path, default=ROOT / 'rebuilt')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.rebuilt_dir, args.results_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
