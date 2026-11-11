#!/usr/bin/env python3
"""Analyze all frozen confirmation cells, including null or opposite results."""
import argparse,json,hashlib,gzip
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from analyze_reader_interventions import parse_names,normalize,csvout,dump,cluster_interval
from revision_controls import interval,metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert json.loads((a.input/'progress.json').read_text())['status']=='PASS'
    manifest=json.loads((a.input/'sample_manifest.json').read_text());sample={r['id']:r for r in manifest['sample']};assert len(sample)==300
    assert not set(sample)&set(manifest['excluded_ids'])
    f=a.input/'records.jsonl.gz'
    if not f.exists(): f=f.with_suffix('')
    with (gzip.open(f,'rt') if f.suffix=='.gz' else f.open()) as h:rows=[json.loads(x) for x in h]
    assert len(rows)==1200;groups=defaultdict(dict)
    for r in rows:
        assert r['id'] in sample and r['prediction']==parse_names(r['raw_response'])
        score=metrics(r['prediction'],set(map(normalize,r['gold_answers'])))
        assert all(abs(score[k]-r['metrics'][k])<1e-12 for k in ('precision','recall','f1'))
        assert r['evidence_tokens']<=256
        key=(r['selector'],r['condition']);assert r['id'] not in groups[key];groups[key][r['id']]=r
    for g in groups.values():assert set(g)==set(sample)
    means=[];per=[]
    for (sel,cond),g in sorted(groups.items()):
        means.append({'selector':sel,'condition':cond,'n':len(g),**{k:float(np.mean([r['metrics'][k] for r in g.values()])) for k in ('precision','recall','f1')},
            'endpoint_f1':float(np.mean([r['endpoint_metrics']['f1'] for r in g.values()])),
            'filtered_f1':float(np.mean([r['filtered_metrics']['f1'] for r in g.values()])),
            'cap_hits':sum(r['hit_generation_cap'] for r in g.values())})
    for qid,src in sorted(sample.items()):
        x={}
        for dec in ('original','no_rep_constraints'):
            full=groups['full',dec][qid];nc=groups['no_cost',dec][qid]
            x['cost_'+dec]=full['metrics']['f1']-nc['metrics']['f1']
        for sel in ('full','no_cost'):
            assert groups[sel,'original'][qid]['prompt']==groups[sel,'no_rep_constraints'][qid]['prompt']
        x['interaction']=x['cost_no_rep_constraints']-x['cost_original']
        x['full_decoder_gain']=groups['full','no_rep_constraints'][qid]['metrics']['f1']-groups['full','original'][qid]['metrics']['f1']
        per.append({'id':qid,'hop':src['hop'],'template':str(src['hop'])+'|'+' '.join(src['question'].casefold().replace(src['topic_entity'].casefold(),'[entity]').split()),**x})
    effects={}
    for name in ('cost_original','cost_no_rep_constraints','interaction','full_decoder_gain'):
        effects[name]={**interval([r[name] for r in per],[r['hop'] for r in per]),'template_cluster':cluster_interval([r[name] for r in per],[r['template'] for r in per])}
    dump(a.output/'paired_effects.json',effects);csvout(a.output/'condition_metrics.csv',means);csvout(a.output/'per_question_effects.csv',per)
    dump(a.output/'verification.json',{'status':'PASS','n_questions':300,'condition_rows':1200,'hop_counts':dict(Counter(r['hop'] for r in per)),
        'template_clusters':len({r['template'] for r in per}),'sample_overlap_ids':0,'reparsed_all_outputs':True,
        'bootstrap_seed':20261110,'bootstrap_resamples':20000,'primary_estimand':'interaction','intervals':'two-sided percentile; paired hop-stratified and whole-template cluster',
        'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'selection_budget_violations':0})
    print(json.dumps(effects,indent=2))

if __name__=='__main__':main()
