#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only post-review scoring audit, NumPy required; no new model inference.
Run: python recompute.py --output results
All new intervals are exploratory, pointwise, paired, stratified by hop/chain length.
"""
from pathlib import Path
from collections import defaultdict
import argparse, csv, json, re, unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=HERE/'results')
ap.add_argument('--inputs',type=Path,default=HERE/'inputs')
ap.add_argument('--verify-original-examples',action='store_true',help='Check archived example predictions; omit for freshly generated answers')
args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
def read(name): return json.loads((args.inputs/name).read_text())
def rows(name): return [json.loads(x) for x in (args.inputs/name).read_text().splitlines()]
def norm(s):
    s=unicodedata.normalize('NFKC',s)
    s=re.sub(r'^\s*(?:answer|answers)\s*:\s*','',s,flags=re.I)
    s=re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',s)
    return ' '.join(s.strip().strip('`"\'[](){}').rstrip('.。').casefold().split())
def f1(pred,gold):
    p,g=set(pred),set(gold)
    return 2*len(p&g)/(len(p)+len(g)) if p or g else 0.
def best_parse(pred,q):
    # Same best-F1 rule as the official evaluator. No eligibility change here;
    # all 1,077 questions pass its Good/Complete question-level gate.
    values=[]
    for p in q['Parses']:
        g=[a['AnswerArgument'] for a in p['Answers']]
        if not g: values.append(1. if not pred else 0.)
        else: values.append(f1(pred,g))
    return max(values)
def ci(d,groups,seed=20261109):
    d=np.asarray(d,dtype=float); groups=np.asarray(groups)
    rng=np.random.default_rng(seed); boot=np.zeros(20000)
    for g in sorted(set(groups)):
        x=d[groups==g]
        for start in range(0,20000,400):
            boot[start:start+400]+=x[rng.integers(0,len(x),size=(400,len(x)))].sum(axis=1)
    return [float(v) for v in np.quantile(boot/len(d),[.025,.975])]
def csvout(name,data):
    with (args.output/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def mean(x): return float(np.mean(x))
def summary(name,data): (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

meta=rows('metaqa_primary.jsonl'); comp=rows('metaqa_components.jsonl')
for r in meta+comp:
    assert abs(f1(r['parsed_predictions'],map(norm,r['gold_answers']))-r['f1'])<1e-12
    assert abs(f1(map(norm,r['selected_endpoints']),map(norm,r['gold_answers']))-r['selected_endpoint_f1'])<1e-12
    assert r['evidence_tokens']<=r['budget']
base={r['id']:r for r in meta if r['budget']==256 and r['condition']=='ra_mmr'}
component_summary=[]
for cond in ['racps_no_relation','racps_no_cost','racps_no_novelty']:
    rr=sorted([r for r in comp if r['condition']==cond],key=lambda r:r['id'])
    assert len(rr)==900 and set(r['id'] for r in rr)==set(base)
    out={'condition':cond,'n':len(rr)}
    for label,k in [('reader_F1','f1'),('endpoint_F1','selected_endpoint_f1'),('endpoint_recall','selected_endpoint_recall')]:
        d=[base[r['id']][k]-r[k] for r in rr]
        out[label]={'full':mean([base[r['id']][k] for r in rr]),'variant':mean([r[k] for r in rr]),'full_minus_variant':mean(d),'ci95':ci(d,[r['hop'] for r in rr])}
    same_set_different_order=[r for r in rr if set(base[r['id']]['selected_evidence'])==set(r['selected_evidence']) and base[r['id']]['selected_evidence']!=r['selected_evidence']]
    out['same_paths_different_order_n']=len(same_set_different_order)
    out['same_paths_different_order_contribution_to_total_reader_difference']=sum(base[r['id']]['f1']-r['f1'] for r in same_set_different_order)/900
    component_summary.append(out)
summary('component_stages.json',component_summary)

meta_diag=[]
for b in [128,256,512]:
    rr=[r for r in meta if r['budget']==b and r['condition']=='ra_mmr']
    extra=[set(r['parsed_predictions'])-set(map(norm,r['selected_endpoints'])) for r in rr]
    meta_diag.append({'budget':b,'n':900,'reader_f1':mean([r['f1'] for r in rr]),
        'endpoint_f1':mean([r['selected_endpoint_f1'] for r in rr]),
        'endpoint_filtered_reader_f1':mean([r['candidate_constrained_f1'] for r in rr]),
        'questions_with_nonendpoint_prediction':sum(bool(x) for x in extra),
        'questions_with_correct_nonendpoint_prediction':sum(bool(x&set(map(norm,r['gold_answers']))) for r,x in zip(rr,extra))})
summary('metaqa_output_diagnostics.json',meta_diag)
csvout('metaqa_primary_means.csv',[{'budget':b,'condition':c,'reader_f1':mean([r['f1'] for r in meta if r['budget']==b and r['condition']==c]),'endpoint_f1':mean([r['selected_endpoint_f1'] for r in meta if r['budget']==b and r['condition']==c])} for b in [128,256,512] for c in ['bfs','bm25','ba_mmr','ra_mmr']])

web=rows('webqsp_new.jsonl'); gold=read('webqsp_parses.json'); scored=[]
assert len(gold)==1077
assert all(any(p.get('AnnotatorComment',{}).get('QuestionQuality')=='Good' and
               p.get('AnnotatorComment',{}).get('ParseQuality')=='Complete'
               for p in q['Parses']) for q in gold.values())
bycond=defaultdict(dict)
for r in web:
    assert r['evidence_tokens']<=256
    assert abs(f1(r['mapped_prediction_ids'],r['gold_ids'])-r['f1'])<1e-12
    assert abs(f1(r['selected_endpoint_ids'],r['gold_ids'])-r['endpoint_metrics']['f1'])<1e-12
    for p in gold[r['id']]['Parses']:
        ids=[a['AnswerArgument'] for a in p['Answers']]
        assert len(ids)==len(set(ids)) # official list scoring equals set scoring
    v={'id':r['id'],'condition':r['condition'],'chain_length':r['annotated_max_hops'],
       'reader_union':r['f1'],'endpoint_union':r['endpoint_metrics']['f1'],
       'reader_best_parse':best_parse(r['mapped_prediction_ids'],gold[r['id']]),
       'endpoint_best_parse':best_parse(r['selected_endpoint_ids'],gold[r['id']])}
    scored.append(v); bycond[r['condition']][r['id']]=v
csvout('webqsp_per_question_scores.csv',scored)
websummary=[]
for c in ['bm25','racps','relation_density','e5_rank','e5_density']:
    rr=sorted(bycond[c].values(),key=lambda r:r['id']); assert len(rr)==1077
    d=[r['reader_best_parse']-r['endpoint_best_parse'] for r in rr]
    websummary.append({'condition':c,'n':len(rr),**{k:mean([r[k] for r in rr]) for k in ['reader_union','endpoint_union','reader_best_parse','endpoint_best_parse']},
        'reader_minus_endpoint_best_parse':mean(d),'increment_ci95':ci(d,[r['chain_length'] for r in rr])})
contrasts=[]
for c in ['bm25','relation_density','e5_rank']:
    ids=sorted(bycond['racps']); d=[bycond['racps'][i]['reader_best_parse']-bycond[c][i]['reader_best_parse'] for i in ids]
    contrasts.append({'contrast':'racps_minus_'+c,'difference':mean(d),'ci95':ci(d,[bycond['racps'][i]['chain_length'] for i in ids])})
valid_different=0
for q in gold.values():
    sets={tuple(sorted(a['AnswerArgument'] for a in p['Answers'])) for p in q['Parses'] if p.get('TopicEntityMid') and p.get('InferentialChain') and p.get('Answers')}
    valid_different+=len(sets)>1
summary('webqsp_scoring_sensitivity.json',{'n':1077,'distinct_valid_parse_answer_sets_questions':valid_different,'seed':20261109,'bootstrap_draws':20000,'status':'exploratory post-review re-scoring; frozen main union analysis retained','conditions':websummary,'contrasts':contrasts})

phi=read('phi_fixed_text_token_counts.json')
summary('phi_token_audit.json',[{'condition':c,'n':len(rr),'over_256':sum(r['reader_tokenizer_evidence_tokens']>256 for r in rr),'max':max(r['reader_tokenizer_evidence_tokens'] for r in rr)} for c in sorted({r['condition'] for r in phi}) for rr in [[r for r in phi if r['condition']==c]]])
# A scorer property must hold for any generated answer, independent of one run.
assert f1(['gold'],['gold','other'])==2/3
assert f1(['gold','wrong'],['gold'])==2/3
if args.verify_original_examples:
    examples=read('recorded_examples.json')
    example=examples['MetaQA']; assert example['id']=='metaqa-2hop-test-02205'
    assert norm('1972') not in set(map(norm,example['selected_endpoints']))
    assert '1972' in set(map(norm,example['gold_answers']))
    assert abs(example['f1']-2/13)<1e-12
    assert f1(['1972'],map(norm,example['gold_answers']))==1/3
else:
    example={'gold_answers':['gold']}
assert f1(['not-a-gold-answer'],map(norm,example['gold_answers']))==0
summary('checks.json',{'metaqa_primary_rows':len(meta),'metaqa_component_rows':len(comp),'webqsp_new_rows':len(web),'saved_primary_scores_match':True,'primary_recorded_caps_pass':True,'outside_endpoint_correct_answer_counts_as_tp':True,'outside_endpoint_wrong_answer_counts_as_fp':True,'official_question_gate_all_1077_pass':True,'no_new_model_calls':True})
print('Recomputed primary means, component stages, output diagnostics, official-rule sensitivity, and Phi counts.')
