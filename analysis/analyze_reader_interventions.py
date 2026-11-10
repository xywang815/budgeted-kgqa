#!/usr/bin/env python3
"""Paired analysis of completed matched reader experiments; no outcome tuning."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,re
from pathlib import Path
import numpy as np
from revision_controls import interval,distribution,metrics


def parse_names(raw):
    text=re.sub(r'<think>.*?</think>','',raw,flags=re.S|re.I).strip()
    return list(dict.fromkeys(v for x in re.split(r'\s*(?:\||;|\n)\s*',text)
                             if (v:=normalize(x)) and v not in {'none','no answer','unknown','n/a'}))


def cluster_interval(values, clusters):
    values=np.asarray(values,float);names=sorted(set(clusters));idx={k:i for i,k in enumerate(names)}
    labels=np.asarray([idx[c] for c in clusters]);sums=np.bincount(labels,weights=values);sizes=np.bincount(labels)
    rng=np.random.default_rng(20261110);boot=[]
    for start in range(0,20000,400):
        take=rng.integers(len(names),size=(min(400,20000-start),len(names)))
        boot.extend((sums[take].sum(1)/sizes[take].sum(1)).tolist())
    return {'difference':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n':len(values),'clusters':len(names)}


def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--meta-reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert json.loads((a.input/'progress.json').read_text())['status']=='PASS'
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    record_path=a.input/'records.jsonl'
    if not record_path.exists():record_path=a.input/'records.jsonl.gz'
    with (gzip.open(record_path,'rt',encoding='utf-8') if record_path.suffix=='.gz' else record_path.open()) as handle:
        rows=[json.loads(line) for line in handle]
    conf=json.loads((a.input/'configuration.json').read_text());assert conf['limit_per_stratum']==0 and len(rows)==18708
    ref=json.loads(a.meta_reference.read_text());topics={r['id']:r['topic_entity'] for r in ref['results']}
    groups=defaultdict(dict);per=[]
    for r in rows:
        k=(r['dataset'],r['selection_budget'],r['selector'],r['condition']);assert r['id'] not in groups[k]
        groups[k][r['id']]=r
        if r['dataset']=='WebQSP' or r['condition'].startswith('labeled_codes'):
            parsed=list(dict.fromkeys(r['answer_code_map'].get(c.lower(),'__invalid_code__'+c.lower()) for c in re.findall(r'(?i)\bA\d+\b',r['raw_response'])))
        else:parsed=parse_names(r['raw_response'])
        assert parsed==r['prediction']
        # Scores independently reconstructed from saved parsed answers.
        gold=r['gold_answers'] if r['dataset']=='WebQSP' else [normalize(x) for x in r['gold_answers']]
        for stage,pred in [('reader',r['prediction']),('filtered',[x for x in r['prediction'] if x in set(r['selected_endpoints'])]),('endpoint',r['selected_endpoints'])]:
            v=metrics(pred,gold);recorded=r['metrics'] if stage=='reader' else r[stage+'_metrics']
            assert all(abs(v[name]-recorded[name])<1e-12 for name in ('precision','recall','f1'))
        if r['condition'] in ('original','no_cost_original'):
            assert r['evidence_tokens']==r['original_evidence_tokens'] and not r['context_over_selection_cap']
        assert len(r['output_ids'])==r['output_tokens']
        per.append({'dataset':r['dataset'],'budget':r['selection_budget'],'selector':r['selector'],'condition':r['condition'],
            'id':r['id'],'stratum':r['stratum'],'f1':r['metrics']['f1'],'precision':r['metrics']['precision'],
            'recall':r['metrics']['recall'],'filtered_f1':r['filtered_metrics']['f1'],'endpoint_f1':r['endpoint_metrics']['f1'],
            'evidence_tokens':r['evidence_tokens'],'total_input_tokens':r['total_input_tokens'],'output_tokens':r['output_tokens'],
            'cap_hit':r['hit_generation_cap'],'context_over_cap':r['context_over_selection_cap']})
    means=[];concordance=[];contrasts=[]
    for k,group in sorted(groups.items()):
        rr=list(group.values());n=len(rr);assert n==(900 if k[0]=='MetaQA' else 1077)
        means.append({'dataset':k[0],'budget':k[1],'selector':k[2],'condition':k[3],'n':n,
            **{m:float(np.mean([r['metrics'][m] for r in rr])) for m in ('precision','recall','f1')},
            'filtered_f1':float(np.mean([r['filtered_metrics']['f1'] for r in rr])),
            'endpoint_f1':float(np.mean([r['endpoint_metrics']['f1'] for r in rr])),
            'cap_hits':sum(r['hit_generation_cap'] for r in rr),'context_over_cap':sum(r['context_over_selection_cap'] for r in rr),
            'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in rr])),
            'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rr])),
            'mean_predicted_answers':float(np.mean([len(set(r['prediction'])) for r in rr]))})
        if k[3] in ('original','no_cost_original'):
            concordance.append({'dataset':k[0],'budget':k[1],'selector':k[2],'condition':k[3],'n':n,
                'raw_match':sum(r['raw_response']==r['historical_raw_response'] for r in rr),
                'f1_match':sum(abs(r['metrics']['f1']-r['historical_f1'])<1e-12 for r in rr),
                'f1_difference':float(np.mean([r['metrics']['f1']-r['historical_f1'] for r in rr]))})
    def template(r):
        text=r['question'].casefold().replace(topics[r['id']].casefold(),'[entity]')
        return str(r['stratum'])+'|'+' '.join(text.split())
    def contrast(dataset,budget,selector,name,left,right,stage='reader',subset=None):
        aa=groups[dataset,budget,selector,left];bb=groups[dataset,budget,selector,right];assert aa.keys()==bb.keys()
        ids=sorted(aa)
        if subset:ids=[i for i in ids if subset(aa[i],bb[i])]
        key='metrics' if stage=='reader' else stage+'_metrics'
        values=[aa[i][key]['f1']-bb[i][key]['f1'] for i in ids];rr=[aa[i] for i in ids]
        result={'dataset':dataset,'budget':budget,'selector':selector,'contrast':name,'stage':stage,
            **interval(values,[r['stratum'] for r in rr])}
        if dataset=='MetaQA':result['template_cluster']=cluster_interval(values,[template(r) for r in rr])
        contrasts.append(result);return result
    for (dataset,budget,selector,condition) in sorted(groups):
        if condition=='original':continue
        contrast(dataset,budget,selector,condition+'_minus_original',condition,'original')
    stage_increments=[]
    for (dataset,budget,selector,condition),group in sorted(groups.items()):
        rr=[group[i] for i in sorted(group)]
        for stage in ('metrics','filtered_metrics'):
            values=[r[stage]['f1']-r['endpoint_metrics']['f1'] for r in rr]
            result={'dataset':dataset,'budget':budget,'selector':selector,'condition':condition,'stage':stage,
                **interval(values,[r['stratum'] for r in rr])}
            if dataset=='MetaQA':result['template_cluster']=cluster_interval(values,[template(r) for r in rr])
            stage_increments.append(result)
    contrast('MetaQA',256,'racps','codes_minus_names_labeled','labeled_codes','labeled_names')
    contrast('MetaQA',256,'racps','codes_minus_names_no_constraints','labeled_codes_no_constraints','labeled_names_no_constraints')
    contrast('MetaQA',256,'racps','codes_minus_names_labeled_filtered','labeled_codes','labeled_names',stage='filtered')
    contrast('MetaQA',256,'racps','codes_minus_names_no_constraints_filtered','labeled_codes_no_constraints','labeled_names_no_constraints',stage='filtered')
    contrast('MetaQA',256,'racps','names_constraints_removed','labeled_names_no_constraints','labeled_names')
    contrast('MetaQA',256,'racps','codes_constraints_removed','labeled_codes_no_constraints','labeled_codes')
    contrast('MetaQA',256,'racps','cost_effect_original','original','no_cost_original')
    contrast('MetaQA',256,'racps','cost_effect_canonical','canonical','no_cost_canonical')
    contrast('MetaQA',256,'racps','cost_effect_canonical_both_cap_compliant','canonical','no_cost_canonical',
             subset=lambda x,y:not x['context_over_selection_cap'] and not y['context_over_selection_cap'])
    g={name:groups['MetaQA',256,'racps',name] for name in ('original','no_cost_original','canonical','no_cost_canonical','labeled_codes','labeled_names','labeled_codes_no_constraints','labeled_names_no_constraints')}
    ids=sorted(g['original']);strata=[g['original'][i]['stratum'] for i in ids];clusters=[template(g['original'][i]) for i in ids]
    def f(name,i):return g[name][i]['metrics']['f1']
    modulation=[(f('original',i)-f('no_cost_original',i))-(f('canonical',i)-f('no_cost_canonical',i)) for i in ids]
    interaction=[(f('labeled_codes',i)-f('labeled_names',i))-(f('labeled_codes_no_constraints',i)-f('labeled_names_no_constraints',i)) for i in ids]
    direct={'order_modulation_original_minus_canonical':{**interval(modulation,strata),'template_cluster':cluster_interval(modulation,clusters)},
            'interface_effect_original_constraints_minus_removed':{**interval(interaction,strata),'template_cluster':cluster_interval(interaction,clusters)}}
    cap_ids=[i for i in ids if not g['canonical'][i]['context_over_selection_cap'] and not g['no_cost_canonical'][i]['context_over_selection_cap']]
    cap_values=[(f('original',i)-f('no_cost_original',i))-(f('canonical',i)-f('no_cost_canonical',i)) for i in cap_ids]
    direct['order_modulation_both_cap_compliant']={**interval(cap_values,[g['original'][i]['stratum'] for i in cap_ids]),
        'template_cluster':cluster_interval(cap_values,[template(g['original'][i]) for i in cap_ids])}
    subsets=defaultdict(list)
    for i in ids:
        x=g['original'][i]['selected_evidence'];y=g['no_cost_original'][i]['selected_evidence']
        name='identical_order' if x==y else 'same_set_different_order' if Counter(x)==Counter(y) else 'different_members'
        subsets[name].append(i)
        assert sorted(x)==g['canonical'][i]['selected_evidence'];assert sorted(y)==g['no_cost_canonical'][i]['selected_evidence']
        if name!='different_members':assert g['canonical'][i]['raw_response']==g['no_cost_canonical'][i]['raw_response']
    direct['order_groups']=[]
    for name,ii in subsets.items():
        direct['order_groups'].append({'group':name,'n':len(ii),
            'original_cost_effect':interval([f('original',i)-f('no_cost_original',i) for i in ii],[g['original'][i]['stratum'] for i in ii]),
            'canonical_cost_effect':interval([f('canonical',i)-f('no_cost_canonical',i) for i in ii],[g['original'][i]['stratum'] for i in ii])})
    dump(a.output/'contrasts.json',contrasts);dump(a.output/'direct_interactions.json',direct);dump(a.output/'historical_concordance.json',concordance)
    dump(a.output/'stage_increments.json',stage_increments)
    stratified=[]
    for k,group in sorted(groups.items()):
        for st in sorted({r['stratum'] for r in group.values()}):
            rr=[r for r in group.values() if r['stratum']==st]
            stratified.append({'dataset':k[0],'budget':k[1],'selector':k[2],'condition':k[3],'stratum':st,'n':len(rr),**{m:float(np.mean([r['metrics'][m] for r in rr])) for m in ('precision','recall','f1')}})
    csvout(a.output/'stratum_metrics.csv',stratified)
    csvout(a.output/'condition_metrics.csv',means);csvout(a.output/'per_question_metrics.csv',per)
    dump(a.output/'analysis_manifest.json',{'seed':20261110,'draws':20000,'scope':'post-review exploratory pointwise; no new confirmatory family',
        'records_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'configuration':conf,'all_score_assertions_passed':True,'condition_rows':len(rows),'new_primary_results_substituted':False})
    print('PASS',len(rows),'condition records;',len(contrasts),'paired contrasts')


def normalize(s):
    import unicodedata
    s=unicodedata.normalize('NFKC',s);s=re.sub(r'^\s*(?:answer|answers)\s*:\s*','',s,flags=re.I);s=re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',s)
    return ' '.join(s.strip().strip('`"\'[](){}').rstrip('.。').casefold().split())

if __name__=='__main__':main()
