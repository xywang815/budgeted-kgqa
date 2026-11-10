#!/usr/bin/env python3
"""Derive the fixed-evidence reader's sampling manifest from a completed reference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new path.')
    data = args.reference.read_bytes()
    ref = json.loads(data)
    assert ref['status'] == 'PASS'
    rows = [r for r in ref['results'] if r['condition'] == 'ra_mmr']
    fields = ('id', 'hop', 'line_number', 'question', 'topic_entity')
    assert len({r['id'] for r in rows}) == 900
    assert all(sum(r['hop'] == h for r in rows) == 300 for h in (1, 2, 3))
    out = {'status': 'PASS', 'created_at': datetime.now(timezone.utc).isoformat(),
           'purpose': 'Fixed sample identifiers derived from the completed Qwen reference',
           'source_result': str(args.reference),
           'source_result_sha256': hashlib.sha256(data).hexdigest(),
           'sampling_seed': ref['configuration']['seed'], 'split': 'test',
           'questions_per_hop': {str(h): 300 for h in (1, 2, 3)},
           'gold_answers_included': False,
           'questions': [{k: r[k] for k in fields} for r in rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {len(rows)} sample identifiers to {args.output}')

if __name__ == '__main__':
    main()
