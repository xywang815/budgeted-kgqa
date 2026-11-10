#!/usr/bin/env python3
"""Audit the complete 2x2 cost/decoder run and estimate direct paired interactions."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from revision_controls import interval,metrics,norm
from analyze_reader_interventions import parse_names,cluster_interval

CONDITIONS=('original','no_cost_original','no_rep_constraints','no_cost_no_constraints')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--meta-reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    state=json.loads((a.input/'progress.json').read_text());assert state['status']=='PASS' and state['completed']==state['expected']==3600
    p=a.input/'records.jsonl';p=p if p.exists() else p.with_suffix('.jsonl.gz')
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:rows=[json.loads(l) for l in f]
    assert len(rows)==3600
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    groups={c:{} for c in CONDITIONS}
    for r in rows:
        g=groups[r['condition']];assert r['id'] not in g;g[r['id']]=r
        predicted=parse_names(r['raw_response']);assert predicted==r['prediction']
        gold=list(map(norm,r['gold_answers']))
        for key,pred in [('metrics',predicted),('endpoint_metrics',r['selected_endpoints'])]:
            v=metrics(pred,gold);assert all(abs(v[k]-r[key][k])<1e-12 for k in ('precision','recall','f1'))
        assert r['evidence_tokens']==r['original_evidence_tokens']<=256 and not r['context_over_selection_cap']
        assert len(r['output_ids'])==r['output_tokens']
        conf=r['generation_configuration'];relaxed=r['condition'] in CONDITIONS[2:]
        assert conf['max_new_tokens']==96 and conf['repetition_penalty']==(1. if relaxed else 1.1) and conf['no_repeat_ngram_size']==(0 if relaxed else 3)
    ids=sorted(groups['original']);assert len(ids)==900 and all(set(g)==set(ids) for g in groups.values())
    topics={r['id']:r['topic_entity'] for r in json.loads(a.meta_reference.read_text())['results']}
    strata=[groups['original'][i]['stratum'] for i in ids]
    clusters=[str(groups['original'][i]['stratum'])+'|'+' '.join(groups['original'][i]['question'].casefold().replace(topics[i].casefold(),'[entity]').split()) for i in ids]
    for i in ids:
        base=groups['original'][i]
        for g in groups.values():
            r=g[i]
            assert r['question']==base['question'] and r['stratum']==base['stratum']
            assert set(map(norm,r['gold_answers']))==set(map(norm,base['gold_answers']))
        for left,right in [('original','no_rep_constraints'),('no_cost_original','no_cost_no_constraints')]:
            assert groups[left][i]['selected_evidence']==groups[right][i]['selected_evidence']
            assert groups[left][i]['prompt']==groups[right][i]['prompt']
    def delta(left,right,field='metrics'):
        return np.array([groups[left][i][field]['f1']-groups[right][i][field]['f1'] for i in ids])
    original=delta('original','no_cost_original');relaxed=delta('no_rep_constraints','no_cost_no_constraints');endpoint=delta('original','no_cost_original','endpoint_metrics')
    effects={key:{**interval(values,strata),'template_cluster':cluster_interval(values,clusters)} for key,values in [
        ('cost_original',original),('cost_no_constraints',relaxed),('decoder_modulation_relaxed_minus_original',relaxed-original),
        ('cost_endpoint',endpoint),('stage_effect_original',original-endpoint),('stage_effect_no_constraints',relaxed-endpoint)]}
    aggregates=[]
    for c,g in groups.items():
        rr=list(g.values());aggregates.append({'condition':c,'n':len(rr),**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in ('precision','recall','f1')},
            'endpoint_f1':float(np.mean([r['endpoint_metrics']['f1'] for r in rr])),'cap_hits':sum(r['hit_generation_cap'] for r in rr),
            'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in rr]))})
    with (a.output/'condition_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]));w.writeheader();w.writerows(aggregates)
    replay=[]
    for condition in CONDITIONS[:2]:
        rr=list(groups[condition].values())
        replay.append({'condition':condition,'n':len(rr),'raw_match':sum(r['raw_response']==r['historical_raw_response'] for r in rr),
            'f1_match':sum(abs(r['metrics']['f1']-r['historical_f1'])<1e-12 for r in rr),
            'f1_delta':float(np.mean([r['metrics']['f1']-r['historical_f1'] for r in rr]))})
    (a.output/'historical_concordance.json').write_text(json.dumps(replay,indent=2)+'\n')
    (a.output/'paired_effects.json').write_text(json.dumps(effects,indent=2)+'\n')
    (a.output/'per_question_effects.json').write_text(json.dumps([{'id':i,'cost_original':float(original[j]),'cost_no_constraints':float(relaxed[j]),'endpoint_cost':float(endpoint[j])} for j,i in enumerate(ids)],indent=2)+'\n')
    receipt={'status':'PASS','records':3600,'raw_parser_and_scores_checked':True,'fixed_path_and_prompt_checks':True,'seed':20261110,'bootstrap_draws':20000,
        'scope':'post-observation exploratory cost/decoder control; pointwise intervals','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_configuration':json.loads((a.input/'configuration.json').read_text()),'records_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (a.output/'analysis_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
