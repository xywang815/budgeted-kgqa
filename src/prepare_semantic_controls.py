#!/usr/bin/env python3
"""CPU-only preparation of frozen MetaQA candidates and WebQSP semantic controls."""
from __future__ import annotations
import argparse,gzip,hashlib,json,random,sys,time
from pathlib import Path
from datetime import datetime,timezone


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def line_costs(tokenizer,texts,labeled):
    if not texts:return []
    lines=['- '+t+(' => A1' if labeled else '') for t in texts]
    return [len(x) for x in tokenizer(lines,add_special_tokens=False)['input_ids']]
def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)
def save_question(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--timeout-hours',type=float,default=8.)
    a=ap.parse_args();p=a.project.resolve();out=a.output.resolve();sys.path.insert(0,str(p/'src'))
    from transformers import AutoTokenizer
    from kgqa_core import load_kg,enumerate_simple_paths,bm25_scores,relation_bm25_scores,rank_by_scores
    from run_webqsp_replication import parse_question,load_question_graph,enumerate_paths,render_labeled_evidence
    started=time.monotonic()
    if out.exists() and not a.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/'questions').mkdir(exist_ok=True)
    sources={'metaqa':p/'results/qwen3_awq_test300_B256.json','web':p/'revision_20260909/results/shared_pool_webqsp_all1577_B256.json',
             'web_train':p/'external/webqsp_official/WebQSP/data/WebQSP.train.json','kg':p/'data/processed/metaqa/kg.tsv'}
    conf={'script_sha256':digest(__file__),'sources_sha256':{k:digest(v) for k,v in sources.items()},'development_seed':20261110,
          'development_sample':50,'development_strata':'annotated max chain ==1 vs >=2','test_tuning':False,
          'candidate_rules':'unchanged primary MetaQA exact known hop cap100000; shared WebQSP top500+top500 fill1000',
          'costs':'Qwen standalone evidence line, including A1 suffix for WebQSP; no special tokens',
          'tuning_choices':['rank','max_density_g05','max_density_g1','minmax_density_g05','minmax_density_g1'],
          'selection_metric':'development endpoint macro F1; ties choose rank then gamma0.5 then gamma1 and max before minmax',
          'scope':'post-review exploratory; rank on MetaQA; development-selected calibration on WebQSP'}
    if a.resume:assert json.loads((out/'configuration.json').read_text())==conf
    else:dump(out/'configuration.json',conf)
    meta=json.loads(sources['metaqa'].read_text());web=json.loads(sources['web'].read_text());train=json.loads(sources['web_train'].read_text())
    assert meta['status']==web['status']=='PASS'
    qwen=AutoTokenizer.from_pretrained(p/'models/Qwen3-8B-AWQ',local_files_only=True)
    graphdir=p/'external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods'
    train_rows=[]
    for q in train['Questions']:
        if not any(z.get('AnnotatorComment',{}).get('QuestionQuality')=='Good' and z.get('AnnotatorComment',{}).get('ParseQuality')=='Complete' for z in q.get('Parses',[])):continue
        r=parse_question(q)
        if r and (graphdir/(r['id']+'.nxhd')).is_file():train_rows.append(r)
    chosen=[]
    for stratum in (1,2):
        rr=sorted([r for r in train_rows if (1 if r['annotated_max_hops']==1 else 2)==stratum],key=lambda r:r['id'])
        random.Random(20261110+stratum).shuffle(rr);n=0
        for r in rr:
            adj,names,_=load_question_graph(graphdir/(r['id']+'.nxhd'))
            if not any(topic in adj for topic in r['topics']):continue
            chosen.append(r);n+=1
            if n==50:break
        assert n==50
    chosen.sort(key=lambda r:r['id']);assert all(r['id'].startswith('WebQTrn-') for r in chosen)
    dump(out/'development_question_ids.json',[r['id'] for r in chosen])
    meta_rows=[r for r in meta['results'] if r['condition']=='ra_mmr'];assert len(meta_rows)==900
    web_rows={r['id']:r for r in web['results'] if r['group']=='extension1077'};assert len(web_rows)==1077
    jobs=[('web_dev',r) for r in chosen]+[('web_test',r) for r in sorted(web_rows.values(),key=lambda r:r['id'])]+[('meta_test',r) for r in meta_rows]
    index=[];adj_meta=None
    for phase,row in jobs:
        if time.monotonic()-started>a.timeout_hours*3600:raise TimeoutError('CPU preparation timeout; caches retained')
        target=out/'questions'/f'{phase}_{row["id"]}.json.gz'
        if a.resume and target.exists():
            with gzip.open(target,'rt') as f:cache=json.load(f)
            assert cache['id']==row['id'] and cache['phase']==phase
        elif phase=='meta_test':
            if adj_meta is None:adj_meta=load_kg(sources['kg'])
            paths,truncated=enumerate_simple_paths(adj_meta,row['topic_entity'],row['hop'],100000,exact_hops_only=True)
            assert len(paths)==row['candidate_paths'] and truncated==row['candidate_truncated']
            texts=[x.text() for x in paths];assert set(row['selected_evidence'])<=set(texts)
            costs=line_costs(qwen,texts,False)
            cache={'id':row['id'],'phase':phase,'question':row['question'],'gold_answers':row['gold_answers'],
                'stratum':row['hop'],'paths':texts,'endpoints':[x.endpoint for x in paths],'hops':[x.hops for x in paths],
                'costs':costs,'candidate_paths':len(paths),'candidate_truncated':truncated,
                'pool_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()}
            save_question(target,cache)
        else:
            adj,names,triples=load_question_graph(graphdir/(row['id']+'.nxhd'))
            paths,endpoints,truncated=enumerate_paths(adj,names,row['topics'],100000)
            full=bm25_scores(row['question'],paths);rel=relation_bm25_scores(row['question'],paths)
            fullorder=rank_by_scores(paths,full);relorder=rank_by_scores(paths,rel)
            pool=list(dict.fromkeys(fullorder[:500]+relorder[:500]));seen=set(pool)
            for x in fullorder:
                if len(pool)>=1000:break
                if x not in seen:pool.append(x);seen.add(x)
            texts=[x.text() for x in pool];pool_hash=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
            if phase=='web_test':assert pool_hash==row['pool_sha256'],row['id']
            costs=line_costs(qwen,texts,True)
            cache={'id':row['id'],'phase':phase,'question':row['question'],'gold_answers':row['gold_ids'],
                'stratum':row['annotated_max_hops'],'paths':texts,'endpoints':[endpoints[x] for x in pool],
                'hops':[x.hops for x in pool],'costs':costs,'candidate_paths':len(paths),'candidate_truncated':truncated,
                'pool_sha256':pool_hash,'subgraph_sha256':digest(graphdir/(row['id']+'.nxhd'))}
            save_question(target,cache)
        index.append({'id':row['id'],'phase':phase,'file':str(target.relative_to(out)),'sha256':digest(target),'pool_paths':len(cache['paths'])})
        if len(index)%10==0:
            dump(out/'progress.json',{'status':'RUNNING','completed':len(index),'expected':len(jobs),'phase':phase,'elapsed_seconds':time.monotonic()-started})
            print('CACHE',len(index),'/',len(jobs),phase,flush=True)
    dump(out/'index.json',index);dump(out/'progress.json',{'status':'PASS','completed':len(index),'expected':len(jobs),'elapsed_seconds':time.monotonic()-started})
    print('PASS',len(index),'candidate caches',flush=True)

if __name__=='__main__':main()
