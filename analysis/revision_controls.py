#!/usr/bin/env python3
"""Post-review analyses of saved outputs. No model calls or test-set tuning.

Inputs are original runner JSON files; outputs remain local. All new intervals
are exploratory paired, stratified bootstrap intervals, not a new Holm family.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np

SEED = 20261110
DRAWS = 20000
META_METHODS = ('bfs', 'bm25', 'ba_mmr', 'ra_mmr')
WEB_METHODS = ('bm25', 'racps', 'relation_density', 'e5_rank', 'e5_density')


def norm(s):
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'^\s*(?:answer|answers)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', s)
    return ' '.join(s.strip().strip('`"\'[](){}').rstrip('.。').casefold().split())


def metrics(pred, gold):
    p, g = set(pred), set(gold)
    n = len(p & g)
    return {'precision': n / len(p) if p else 0.,
            'recall': n / len(g) if g else 0.,
            'f1': 2 * n / (len(p) + len(g)) if p or g else 0.}


def path_endpoints(row, web=False):
    if web:
        result = [row['answer_code_map']['a' + str(i + 1)]
                  for i in range(len(row['selected_evidence']))]
        assert set(result) == set(row['selected_endpoint_ids'])
    else:
        result = [norm(re.split(r'(?:--\[[^\]]+\]-->|<--\[[^\]]+\]--)\s*', p)[-1])
                  for p in row['selected_evidence']]
        assert set(result) == set(map(norm, row['selected_endpoints'])), row['id']
    assert len(result) == row['selected_paths']
    return result


def vote_answers(endpoints, rule):
    counts = Counter(endpoints)
    if not counts:
        return []
    if rule == 'max_vote':
        return [e for e in counts if counts[e] == max(counts.values())]
    if rule == 'vote_top5':
        # Stable sorting preserves first appearance for equal vote counts.
        return sorted(counts, key=lambda e: -counts[e])[:5]
    raise ValueError(rule)


def interval(values, strata):
    values, strata = np.asarray(values, float), np.asarray(strata)
    assert len(values) == len(strata) and len(values)
    rng = np.random.default_rng(SEED)
    boot = np.zeros(DRAWS)
    for group in sorted(set(strata)):
        x = values[strata == group]
        for start in range(0, DRAWS, 400):
            take = min(400, DRAWS - start)
            boot[start:start + take] += x[rng.integers(len(x), size=(take, len(x)))].sum(axis=1)
    return {'difference': float(values.mean()),
            'ci95': np.quantile(boot / len(values), [.025, .975]).tolist(), 'n': len(values)}


def distribution(values):
    x = np.asarray(values, float)
    return {'mean': float(x.mean()), 'p10': float(np.quantile(x, .1)),
            'median': float(np.median(x)), 'p90': float(np.quantile(x, .9)),
            'min': float(x.min()), 'max': float(x.max())}


def jsonout(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def run(a):
    out = a.output
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; preserve prior analyses.')
    (out / 'records').mkdir(parents=True, exist_ok=True)
    sources = {}

    def load(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    if a.records_dir:
        def read_records(name):
            path = a.records_dir / (name + '.jsonl')
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            return [json.loads(line) for line in path.read_text().splitlines()]
        meta = read_records('metaqa_primary')
        comp = read_records('metaqa_components')
        web = read_records('webqsp_new')
        webfile = {'questions': list({r['id']: r for r in web}.values())}
    else:
        if not a.metaqa or not a.components or not a.webqsp:
            raise ValueError('Supply --records-dir or all original runner inputs.')
        meta = [r for f in a.metaqa for r in load(f)['results'] if r['condition'] in META_METHODS]
        comp = load(a.components)['results']
        webfile = load(a.webqsp)
        web = [r for r in webfile['results'] if r['group'] == 'extension1077']
    base = {r['id']: r for r in meta if r['budget'] == 256 and r['condition'] == 'ra_mmr'}
    assert len(base) == 900 and len(meta) == 10800 and len(web) == 5385
    gold = load(a.webqsp_official)
    official = {q['QuestionId']: q for q in gold['Questions']} if 'Questions' in gold else gold
    per_question, contexts, contrasts, meta_interface = [], [], [], []

    def process(rows, dataset, web_mode=False):
        if not rows:
            return
        groups = defaultdict(list)
        for r in rows:
            endpoints = path_endpoints(r, web_mode)
            pred = r['mapped_prediction_ids'] if web_mode else r['parsed_predictions']
            gold = r['gold_ids'] if web_mode else list(map(norm, r['gold_answers']))
            alternatives = {'endpoint': list(dict.fromkeys(endpoints)), 'reader': pred,
                            'filtered_reader': [x for x in pred if x in set(endpoints)],
                            'max_vote': vote_answers(endpoints, 'max_vote'),
                            'vote_top5': vote_answers(endpoints, 'vote_top5')}
            vals = {k: metrics(v, gold) for k, v in alternatives.items()}
            assert abs(vals['reader']['f1'] - r['f1']) < 1e-12
            expected = r['endpoint_metrics']['f1'] if web_mode else r['selected_endpoint_f1']
            assert abs(vals['endpoint']['f1'] - expected) < 1e-12
            if not web_mode:
                assert abs(vals['filtered_reader']['f1'] - r['candidate_constrained_f1']) < 1e-12
            group = r['annotated_max_hops'] if web_mode else r['hop']
            budget = 256 if web_mode else r['budget']
            row = {'dataset': dataset, 'id': r['id'], 'condition': r['condition'],
                   'budget': budget, 'stratum': group,
                   **{f'{k}_{metric}': value for k, v in vals.items() for metric, value in v.items()}}
            if web_mode:
                q = official[r['id']]
                assert any(x['AnnotatorComment']['QuestionQuality'] == 'Good' and
                           x['AnnotatorComment']['ParseQuality'] == 'Complete' for x in q['Parses'])
                for k, predset in alternatives.items():
                    fs = []
                    for parse in q['Parses']:
                        gg = [x['AnswerArgument'] for x in parse['Answers']]
                        fs.append(metrics(predset, gg)['f1'] if gg else float(not predset))
                    row[k + '_best_parse_f1'] = max(fs)
            per_question.append(row)
            groups[(budget, r['condition'])].append((r, row))
        for (budget, method), pairs in sorted(groups.items()):
            rr, vv = zip(*pairs)
            strata = [v['stratum'] for v in vv]
            for alternative in ['endpoint', 'max_vote', 'vote_top5']:
                contrasts.append({'dataset': dataset, 'budget': budget, 'condition': method,
                                  'contrast': 'reader_minus_' + alternative,
                                  **interval([v['reader_f1'] - v[alternative + '_f1'] for v in vv], strata)})
            if not web_mode and dataset == 'MetaQA' and method == 'ra_mmr':
                meta_interface.append({'budget': budget, 'raw': float(np.mean([v['reader_f1'] for v in vv])),
                                       'filtered': float(np.mean([v['filtered_reader_f1'] for v in vv])),
                                       'endpoint': float(np.mean([v['endpoint_f1'] for v in vv])),
                                       'raw_minus_endpoint': interval([v['reader_f1'] - v['endpoint_f1'] for v in vv], strata),
                                       'filtered_minus_endpoint': interval([v['filtered_reader_f1'] - v['endpoint_f1'] for v in vv], strata),
                                       'filtered_minus_raw': interval([v['filtered_reader_f1'] - v['reader_f1'] for v in vv], strata)})
            h = [len(re.findall(r'(?:--\[[^\]]+\]-->|<--\[[^\]]+\]--)', s)) for r in rr for s in r['selected_evidence']]
            contexts.append({'dataset': dataset, 'budget': budget, 'condition': method, 'n': len(rr),
                             'selected_paths': distribution([r['selected_paths'] for r in rr]),
                             'distinct_endpoints': distribution([len(set(path_endpoints(r, web_mode))) for r in rr]),
                             'budget_utilization': distribution([r['evidence_tokens'] / budget for r in rr]),
                             'output_tokens': distribution([r['output_tokens'] for r in rr]),
                             'predicted_answers': distribution([len(set(r['mapped_prediction_ids'] if web_mode else r['parsed_predictions'])) for r in rr]),
                             'gold_answers': distribution([len(set(r['gold_ids'] if web_mode else map(norm, r['gold_answers']))) for r in rr]),
                             'hit_generation_cap': sum(r['output_hit_max_new_tokens'] for r in rr),
                             'candidate_truncated': sum(r['candidate_truncated'] for r in rr),
                             'selected_twohop_path_fraction': sum(x == 2 for x in h) / len(h) if h else 0.,
                             'selected_hops': dict(Counter(h)),
                             'retrieval_seconds': float(np.mean([r['method_retrieval_seconds'] if web_mode else r['retrieval_seconds'] for r in rr])),
                             'generation_seconds': float(np.mean([r['generation_seconds'] for r in rr]))})
    for r in comp:
        endpoints = path_endpoints(r)
        gold = list(map(norm, r['gold_answers']))
        assert abs(metrics(r['parsed_predictions'], gold)['f1'] - r['f1']) < 1e-12
        assert abs(metrics(endpoints, gold)['f1'] - r['selected_endpoint_f1']) < 1e-12
    process(meta, 'MetaQA')
    process(web, 'WebQSP', True)
    stages = []
    for method in ['racps_no_relation', 'racps_no_cost', 'racps_no_novelty']:
        rr = sorted((r for r in comp if r['condition'] == method), key=lambda r: r['id'])
        assert len(rr) == 900 and {r['id'] for r in rr} == set(base)
        delta = [(base[r['id']]['f1'] - r['f1']) -
                 (base[r['id']]['selected_endpoint_f1'] - r['selected_endpoint_f1']) for r in rr]
        buckets = defaultdict(list)
        for r in rr:
            b = base[r['id']]
            key = ('identical_order' if b['selected_evidence'] == r['selected_evidence'] else
                   'same_set_different_order' if Counter(b['selected_evidence']) == Counter(r['selected_evidence']) else 'different_members')
            buckets[key].append((b, r))
        decomposition = []
        for k, pairs in buckets.items():
            ds = [b['f1'] - r['f1'] for b, r in pairs]
            decomposition.append({'group': k, **interval(ds, [r['hop'] for _, r in pairs]),
                                  'contribution_to_total': sum(ds) / 900,
                                  'different_raw_outputs': sum(b['raw_response'] != r['raw_response'] for b, r in pairs)})
        stages.append({'condition': method, 'reader_minus_endpoint_effect': interval(delta, [r['hop'] for r in rr]),
                       'selection_groups': decomposition})
    jsonout(out / 'component_effect_difference.json', stages)
    jsonout(out / 'output_interface.json', meta_interface)
    jsonout(out / 'reader_comparators.json', contrasts)
    jsonout(out / 'context_and_generation.json', contexts)
    questions = [q for q in webfile['questions'] if q['group'] == 'extension1077']
    jsonout(out / 'candidate_coverage.json', {'n': len(questions),
        'enumeration_cap_hits': sum(q['candidate_truncated'] for q in questions),
        'candidate_paths': distribution([q['candidate_paths'] for q in questions]),
        'proposal_paths': distribution([q['common_pool_paths'] for q in questions]),
        'enumerated_endpoint_recall': float(np.mean([q['candidate_recall'] for q in questions])),
        'proposal_endpoint_recall': float(np.mean([q['common_pool_recall'] for q in questions]))})
    csvout(out / 'metaqa_per_question.csv', [r for r in per_question if r['dataset'] == 'MetaQA'])
    csvout(out / 'webqsp_per_question.csv', [r for r in per_question if r['dataset'] == 'WebQSP'])
    means = []
    for dataset in ['MetaQA', 'WebQSP']:
        rows = [r for r in per_question if r['dataset'] == dataset]
        for budget, method in sorted({(r['budget'], r['condition']) for r in rows}):
            for group in ['all'] + sorted({str(r['stratum']) for r in rows}):
                vv = [r for r in rows if r['budget'] == budget and r['condition'] == method and
                      (group == 'all' or str(r['stratum']) == group)]
                for stage in ['endpoint', 'reader', 'filtered_reader', 'max_vote', 'vote_top5']:
                    means.append({'dataset': dataset, 'budget': budget, 'condition': method,
                                  'stratum': group, 'stage': stage, 'n': len(vv),
                                  **{k: float(np.mean([r[stage + '_' + k] for r in vv])) for k in ['precision', 'recall', 'f1']},
                                  'best_parse_f1': float(np.mean([r[stage + '_best_parse_f1'] for r in vv])) if dataset == 'WebQSP' else ''})
    csvout(out / 'complete_main_metrics.csv', means)
    # The reviewer archive includes raw, parsed, and selected-path records, not weights/graphs.
    for name, rows in [('metaqa_primary', meta), ('metaqa_components', comp), ('webqsp_new', web)]:
        with (out / 'records' / (name + '.jsonl')).open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
    jsonout(out / 'analysis_manifest.json', {'seed': SEED, 'bootstrap_draws': DRAWS,
        'status': 'post-review exploratory; original primary results unchanged',
        'new_model_calls': 0, 'sources_sha256': sources,
        'rule_selection': 'fixed max-vote with all ties; literature-fixed top-5 vote with stable ties; no test tuning',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'missing_new_inference': ['coded MetaQA', 'canonical order', 'larger output cap', 'decoding penalties removed', 'MetaQA E5']})
    print('Saved-score assertions and exact endpoint extraction passed; post-review analyses written to', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', type=Path, help='Alternative: extracted records/ from the reviewer archive')
    parser.add_argument('--metaqa', type=Path, action='append')
    parser.add_argument('--components', type=Path)
    parser.add_argument('--webqsp', type=Path)
    parser.add_argument('--webqsp-official', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
