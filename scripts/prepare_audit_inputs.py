#!/usr/bin/env python3
"""Extract CPU-audit inputs from local experiment outputs and official WebQSP data."""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--output', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'analysis/audit/inputs')
    args = ap.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing inputs: {out}')
    def read(path):
        data=json.loads((root / path).read_text())
        if path!='data/WebQSP.test.json' and data.get('status')!='PASS':
            raise ValueError(f'Incomplete experiment: {path}')
        return data
    meta = [r for b in (128, 256, 512)
            for r in read(f'results/qwen3_awq_test300_B{b}.json')['results']
            if r['condition'] in ('bfs', 'bm25', 'ba_mmr', 'ra_mmr')]
    comp = [r for r in read('results/clean_component_ablation_qwen_B256.json')['results']
            if r['condition'] in ('racps_no_relation', 'racps_no_cost', 'racps_no_novelty')]
    web = [r for r in read('supplement/results/shared_pool_webqsp_all1577_B256.json')['results']
           if r['group'] == 'extension1077']
    ids = {r['id'] for r in web}
    questions = {q['QuestionId']: q for q in read('data/WebQSP.test.json')['Questions']
                 if q['QuestionId'] in ids}
    phi = read('results/week2_shared_evidence_phi_qwenpaths_test300_B256.json')['results']
    example = next(r for r in meta if r['id'] == 'metaqa-2hop-test-02205'
                   and r['condition'] == 'ra_mmr' and r['budget'] == 256)
    assert len(meta) == 10800 and len(comp) == 2700 and len(web) == 5385
    assert len({(r['budget'],r['id'],r['condition']) for r in meta})==len(meta)
    assert len({(r['id'],r['condition']) for r in comp})==len(comp)
    assert len({(r['id'],r['condition']) for r in web})==len(web)
    assert len(questions) == len(ids) == 1077
    out.mkdir(parents=True)
    for name, rows in [('metaqa_primary', meta), ('metaqa_components', comp), ('webqsp_new', web)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    for name, data in [('webqsp_parses', questions), ('phi_fixed_text_token_counts', phi),
                       ('recorded_examples', {'MetaQA': example})]:
        (out / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared local CPU audit inputs at {out}; no upload or model calls.')

if __name__ == '__main__':
    main()
