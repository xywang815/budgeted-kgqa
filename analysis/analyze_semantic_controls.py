#!/usr/bin/env python3
"""Audit the complete semantic-control run and export paired, exploratory results."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from revision_controls import interval,metrics,norm,distribution,vote_answers
from analyze_reader_interventions import cluster_interval,parse_names


def read_lines(path):
    if not path.exists():path=path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        return [json.loads(line) for line in f]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--meta-reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();status=json.loads((a.input/'progress.json').read_text())
    assert status['status']=='PASS' and status['completed_reader_rows']==8631 and status['completed_pools']==2077
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    selected=read_lines(a.input/'selection_records.jsonl');rows=read_lines(a.input/'reader_records.jsonl')
    assert len(selected)==2077 and len(rows)==8631
    choice=json.loads((a.input/'development_choice.json').read_text())
    config=json.loads((a.input/'configuration.json').read_text())
    assert choice['created_before_test_scoring'] and not choice['test_results_used']
    topics={r['id']:r['topic_entity'] for r in json.loads(a.meta_reference.read_text())['results']}
    pools={};diagnostics=[];dev=defaultdict(list);byrule=defaultdict(list);array_summaries=defaultdict(list)
    for row in selected:
        key=(row['phase'],row['id']);assert key not in pools;pools[key]=row
        path=a.input/'arrays'/row['arrays'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['arrays_sha256']
        with np.load(path) as z:scores=z['cosine'];cost=z['cost'];hops=z['hops']
        assert len(scores)==len(cost)==len(hops)==row['pool_paths']
        assert np.isfinite(scores).all() and (cost>0).all()
        gold=row['gold_answers'] if row['phase']!='meta_test' else list(map(norm,row['gold_answers']))
        if len(scores):
            array_summaries[row['phase']].append({'min_cosine':float(scores.min()),'max_cosine':float(scores.max()),
                'median_cosine':float(np.median(scores)),'dynamic_range':float(np.ptp(scores)),
                'range_over_max':float(np.ptp(scores)/scores.max()) if scores.max()>0 else 0.,
                'median_cost':float(np.median(cost)),'p90_cost':float(np.quantile(cost,.9))})
        for rule,s in row['selections'].items():
            ii=np.asarray(s['indices'],int);assert len(set(ii.tolist()))==len(ii)
            assert all(0<=i<len(scores) for i in ii) and s['evidence_tokens']<=256
            assert s['selected_hops']==hops[ii].tolist()
            v=metrics(s['endpoints'],gold)
            assert all(abs(v[k]-s['endpoint_metrics'][k])<1e-12 for k in ('precision','recall','f1'))
            if row['phase']=='web_dev':dev[rule].append(v['f1'])
            base={'phase':row['phase'],'id':row['id'],'stratum':row['stratum'],'rule':rule,
                'pool_paths':len(scores),'pool_two_hop':int((hops==2).sum()),'selected_paths':len(ii),
                'selected_two_hop':int((hops[ii]==2).sum()),'selected_unique_endpoints':len(set(s['endpoints'])),
                'evidence_tokens':s['evidence_tokens'],'endpoint_f1':v['f1'],
                'selected_mean_cosine':float(scores[ii].mean()) if len(ii) else 0.,
                'selected_mean_cost':float(cost[ii].mean()) if len(ii) else 0.,
                'candidate_truncated':row['candidate_truncated']}
            diagnostics.append(base);byrule[row['phase'],rule].append(base)
    means={c:float(np.mean(dev[c])) for c in config['choices']};assert all(len(dev[c])==100 for c in config['choices'])
    assert all(abs(means[c]-choice['development_endpoint_macro_f1'][c])<1e-12 for c in means)
    assert max(config['choices'],key=lambda c:(means[c],-config['choices'].index(c)))==choice['rule']
    groups=defaultdict(dict);per=[]
    for r in rows:
        k=(r['dataset'],r['selector'],r['condition']);assert r['id'] not in groups[k];groups[k][r['id']]=r
        web=r['dataset']=='WebQSP';gold=r['gold_answers'] if web else list(map(norm,r['gold_answers']))
        if web:
            labels={f'a{i+1}':x for i,x in enumerate(r['endpoints'])}
            predicted=list(dict.fromkeys(labels.get(c.lower(),'__invalid_code__'+c.lower()) for c in re.findall(r'(?i)\bA\d+\b',r['raw_response'])))
        else:predicted=parse_names(r['raw_response'])
        assert predicted==r['prediction']
        for field,pred in [('metrics',predicted),('endpoint_metrics',r['endpoints'])]:
            calc=metrics(pred,gold);assert all(abs(calc[k]-r[field][k])<1e-12 for k in ('precision','recall','f1'))
        assert len(r['output_ids'])==r['output_tokens'] and r['evidence_tokens']<=256
        endpoints=r['endpoints'] if web else [norm(re.split(r'(?:--\[[^\]]+\]-->|<--\[[^\]]+\]--)\s*',p)[-1]) for p in r['paths']]
        assert set(endpoints)==set(r['endpoints'])
        r['vote_metrics']={rule:metrics(vote_answers(endpoints,rule),gold) for rule in ('max_vote','vote_top5')}
        per.append({'dataset':r['dataset'],'selector':r['selector'],'condition':r['condition'],'rule':r['rule'],'id':r['id'],'stratum':r['stratum'],
            **{m:r['metrics'][m] for m in ('precision','recall','f1')},'endpoint_f1':r['endpoint_metrics']['f1'],
            **{rule+'_f1':r['vote_metrics'][rule]['f1'] for rule in ('max_vote','vote_top5')},
            'evidence_tokens':r['evidence_tokens'],'cap_hit':r['hit_generation_cap'],'output_tokens':r['output_tokens']})
    aggregate=[];concordance=[]
    for k,g in sorted(groups.items()):
        rr=list(g.values());assert len(rr)==(900 if k[0]=='MetaQA' else 1077)
        for stratum in ['all',*sorted({r['stratum'] for r in rr})]:
            part=rr if stratum=='all' else [r for r in rr if r['stratum']==stratum]
            aggregate.append({'dataset':k[0],'selector':k[1],'condition':k[2],'stratum':stratum,'n':len(part),
                **{m:float(np.mean([r['metrics'][m] for r in part])) for m in ('precision','recall','f1')},
                'endpoint_f1':float(np.mean([r['endpoint_metrics']['f1'] for r in part])),
                **{rule+'_f1':float(np.mean([r['vote_metrics'][rule]['f1'] for r in part])) for rule in ('max_vote','vote_top5')},
                'mean_evidence_tokens':float(np.mean([r['evidence_tokens'] for r in part])),
                'mean_output_tokens':float(np.mean([r['output_tokens'] for r in part])),
                'cap_hits':sum(r['hit_generation_cap'] for r in part)})
        if k[2]=='original' and 'historical_f1' in rr[0]:
            concordance.append({'dataset':k[0],'selector':k[1],'condition':k[2],'n':len(rr),
                'raw_match':sum(r['raw_response']==r['historical_raw_response'] for r in rr),
                'f1_match':sum(abs(r['metrics']['f1']-r['historical_f1'])<1e-12 for r in rr),
                'f1_delta':float(np.mean([r['metrics']['f1']-r['historical_f1'] for r in rr]))})
    contrasts=[]
    comparisons=[('MetaQA','e5_rank','bm25'),('MetaQA','e5_rank','racps'),('MetaQA','racps','bm25'),
                 ('WebQSP','e5_development_choice','e5_rank'),('WebQSP','e5_development_choice','e5_density'),('WebQSP','e5_rank','e5_density')]
    for dataset,left,right in comparisons:
      for condition in (('original','no_rep_constraints') if dataset=='MetaQA' else ('original',)):
        aa=groups[dataset,left,condition];bb=groups[dataset,right,condition];assert aa.keys()==bb.keys();ids=sorted(aa)
        for stage in ('metrics','endpoint_metrics'):
            values=[aa[i][stage]['f1']-bb[i][stage]['f1'] for i in ids]
            result={'dataset':dataset,'condition':condition,'contrast':left+'_minus_'+right,'stage':stage,**interval(values,[aa[i]['stratum'] for i in ids])}
            if dataset=='MetaQA':
                clusters=[str(aa[i]['stratum'])+'|'+' '.join(aa[i]['question'].casefold().replace(topics[i].casefold(),'[entity]').split()) for i in ids]
                result['template_cluster']=cluster_interval(values,clusters)
            contrasts.append(result)
    decoder_effects=[]
    for selector in ('bm25','racps','e5_rank'):
        aa=groups['MetaQA',selector,'no_rep_constraints'];bb=groups['MetaQA',selector,'original'];ids=sorted(aa)
        values=[aa[i]['metrics']['f1']-bb[i]['metrics']['f1'] for i in ids]
        clusters=[str(aa[i]['stratum'])+'|'+' '.join(aa[i]['question'].casefold().replace(topics[i].casefold(),'[entity]').split()) for i in ids]
        decoder_effects.append({'selector':selector,**interval(values,[aa[i]['stratum'] for i in ids]),'template_cluster':cluster_interval(values,clusters)})
    dump(a.output/'decoder_effects.json',decoder_effects)
    interactions=[]
    for left,right in [('e5_rank','racps'),('racps','bm25'),('e5_rank','bm25')]:
        ids=sorted(groups['MetaQA',left,'original'])
        def gain(condition,i):return groups['MetaQA',left,condition][i]['metrics']['f1']-groups['MetaQA',right,condition][i]['metrics']['f1']
        values=[gain('no_rep_constraints',i)-gain('original',i) for i in ids];rr=[groups['MetaQA',left,'original'][i] for i in ids]
        clusters=[str(r['stratum'])+'|'+' '.join(r['question'].casefold().replace(topics[r['id']].casefold(),'[entity]').split()) for r in rr]
        interactions.append({'contrast':left+'_minus_'+right,'interaction':'gain without repetition constraints minus original gain',
            **interval(values,[r['stratum'] for r in rr]),'template_cluster':cluster_interval(values,clusters)})
    dump(a.output/'selector_decoder_interactions.json',interactions)
    reader_increments=[]
    for (dataset,selector,condition),group in sorted(groups.items()):
        rr=[group[i] for i in sorted(group)]
        for comparator in ('endpoint','max_vote','vote_top5'):
            baseline=[r['endpoint_metrics']['f1'] if comparator=='endpoint' else r['vote_metrics'][comparator]['f1'] for r in rr]
            values=[r['metrics']['f1']-b for r,b in zip(rr,baseline)]
            v={'dataset':dataset,'selector':selector,'condition':condition,'comparator':comparator,**interval(values,[r['stratum'] for r in rr])}
            if dataset=='MetaQA':
                clusters=[str(r['stratum'])+'|'+' '.join(r['question'].casefold().replace(topics[r['id']].casefold(),'[entity]').split()) for r in rr]
                v['template_cluster']=cluster_interval(values,clusters)
            reader_increments.append(v)
    dump(a.output/'reader_increments.json',reader_increments)
    summary=[]
    for (phase,rule),rr in sorted(byrule.items()):
        two=sum(r['selected_two_hop'] for r in rr);paths=sum(r['selected_paths'] for r in rr);pooltwo=sum(r['pool_two_hop'] for r in rr)
        summary.append({'phase':phase,'rule':rule,'n':len(rr),'selected_two_hop_share':two/paths if paths else 0.,
            'two_hop_candidate_retention':two/pooltwo if pooltwo else 0.,'selected_two_hop_paths':two,'selected_paths':paths,'pool_two_hop_paths':pooltwo,
            **{'mean_'+key:float(np.mean([r[key] for r in rr])) for key in ('selected_paths','selected_unique_endpoints','evidence_tokens','selected_mean_cosine','selected_mean_cost','endpoint_f1')}})
    distributions={phase:{key:distribution([r[key] for r in rr]) for key in rr[0]} for phase,rr in array_summaries.items()}
    agreement={rule:sum(r['historical_selection_agreement'][rule] for r in selected if r['phase']=='web_test') for rule in ('rank','max_density_g1')}
    dump(a.output/'paired_contrasts.json',contrasts);dump(a.output/'development_choice.json',choice)
    dump(a.output/'historical_concordance.json',{'reader':concordance,'web_exact_selection_counts_of_1077':agreement})
    dump(a.output/'candidate_distributions.json',{'weighting':'Each statistic first computed within a nonempty question pool; distribution across questions. Empty pools retained in all metrics.',
        'pool_counts':{phase:{'all':sum(r['phase']==phase for r in selected),'nonempty':len(rr),
            'encoder_truncated_texts':sum(r['encoder_truncated_texts'] for r in selected if r['phase']==phase)}
            for phase,rr in array_summaries.items()},'phases':distributions})
    csvout(a.output/'condition_metrics.csv',aggregate);csvout(a.output/'per_question_metrics.csv',per)
    csvout(a.output/'pool_diagnostics.csv',diagnostics);csvout(a.output/'selection_summary.csv',summary)
    dump(a.output/'analysis_manifest.json',{'seed':20261110,'bootstrap_draws':20000,'scope':'post-review exploratory pointwise; no new confirmatory family',
        'rows':len(rows),'pools':len(selected),'raw_parser_and_scores_checked':True,'candidate_arrays_hashes_checked':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'configuration':config})
    print('PASS',len(rows),'reader records and',len(selected),'scored candidate pools')


if __name__=='__main__':main()
