#!/usr/bin/env python3
"""SUPP-20260909. Frozen shared-pool selection and exact labeled-budget audit."""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from importlib.metadata import version as package_version
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from kgqa_core import bm25_scores, relation_bm25_scores, rank_by_scores, _jaccard
from run_kgqa_inference import atomic_json, running_gpu_compute_processes
from run_webqsp_replication import (parse_question, load_question_graph, enumerate_paths,
    render_labeled_evidence, build_labeled_prompt, id_metrics)
CONDITIONS=['bm25','racps','relation_density','e5_rank','e5_density']
E5_REV='ffb93f3bd4047442299a41ebb6fa998a38507c52'
QWEN_REV='cb7d6a337aadb4d2082ed0dcef1032e4f8645194'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def count(tokenizer,text): return len(tokenizer.encode(text,add_special_tokens=False))
def strict_codes(response, labels):
    return list(dict.fromkeys(labels.get(c.lower(),'__invalid_code__'+c.lower())
        for c in re.findall(r'(?i)\bA\d+\b',response)))
def select(paths, scores, endpoints, tokenizer, budget, condition, costs, signatures):
    """Stable greedy density or rank, rechecking the full labeled serialization."""
    maximum=max(scores,default=0.)
    rel=[max(0.,s)/maximum if maximum>0 else 0. for s in scores]
    remaining=set(range(len(paths))); chosen=[]; chosen_indices=[]
    novelty=[1.]*len(paths)
    while remaining:
        keys=[]
        for i in remaining:
            if condition=='racps': utility=(.9*rel[i]+.1*novelty[i])/costs[i]
            elif condition in ('relation_density','e5_density'): utility=rel[i]/costs[i]
            else: utility=scores[i]
            # Cost tie-break matches RACPS for density methods; lexical rank is stable.
            key=(utility,rel[i],-costs[i],-i) if condition in ('racps','relation_density','e5_density') else (utility,-i)
            keys.append((key,i))
        accepted=None
        for _,i in sorted(keys,reverse=True):
            evidence,_=render_labeled_evidence(chosen+[paths[i]],endpoints)
            if count(tokenizer,evidence)<=budget:
                accepted=i;break
        # No candidate currently fits; do not assert monotonic tokenizer costs.
        if accepted is None: break
        chosen.append(paths[accepted]);chosen_indices.append(accepted);remaining.remove(accepted)
        if condition=='racps':
            for i in remaining: novelty[i]=min(novelty[i],1.-_jaccard(signatures[i],signatures[accepted]))
    return chosen,chosen_indices

def encode(texts, tokenizer, model, batch_size):
    chunks=[];truncated=0
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        lengths=tokenizer(batch,truncation=False,add_special_tokens=True,return_length=True)['length']
        truncated+=sum(x>512 for x in lengths)
        tok=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            hidden=model(**tok).last_hidden_state
            hidden=hidden.masked_fill(~tok['attention_mask'][...,None].bool(),0.)
            pooled=hidden.sum(dim=1)/tok['attention_mask'].sum(dim=1)[...,None]
            chunks.append(torch.nn.functional.normalize(pooled,p=2,dim=1))
    return torch.cat(chunks) if chunks else torch.empty((0,384),device='cuda'),truncated

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();project=a.project.resolve();a.output=a.output.resolve()
    if a.output.exists() and not a.resume:raise RuntimeError('Output exists, require explicit --resume')
    if running_gpu_compute_processes(): raise RuntimeError('GPU in use, no other process will be stopped')
    random.seed(20260909);torch.manual_seed(20260909);torch.cuda.manual_seed_all(20260909)
    torch.set_num_threads(8);torch.cuda.set_device(0)
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(project/'models/Qwen3-8B-AWQ',local_files_only=True)
    e5tok=AutoTokenizer.from_pretrained(project/'revision_20260909/models/e5-small-v2',local_files_only=True)
    e5=AutoModel.from_pretrained(project/'revision_20260909/models/e5-small-v2',local_files_only=True,torch_dtype=torch.float32).to('cuda').eval()
    model=AutoModelForCausalLM.from_pretrained(project/'models/Qwen3-8B-AWQ',local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    load_seconds=time.perf_counter()-start
    if a.smoke:
        from kgqa_core import Edge,GraphPath
        p1=GraphPath(('Ada','UK'),(Edge('Ada','nationality','UK'),))
        p2=GraphPath(('Ada','London'),(Edge('Ada','birth_place','London'),))
        paths=[p1,p2];endpoints={p1:'country_uk',p2:'city_london'}
        costs=[count(tokenizer,render_labeled_evidence([p],endpoints)[0]) for p in paths]
        sigs=[p.directional_signature() for p in paths]
        checks=[]
        for b in [1,16,32,256]:
            for c in CONDITIONS:
                selected,_=select(paths,[.8,.5],endpoints,tokenizer,b,c,costs,sigs)
                ev,labs=render_labeled_evidence(selected,endpoints);assert count(tokenizer,ev)<=b
                checks.append((b,c,count(tokenizer,ev)))
        assert id_metrics(strict_codes('A1 | A999',{'a1':'gold'}),['gold'])['f1']==2/3
        assert id_metrics(strict_codes('A999',{'a1':'gold'}),['gold'])['f1']==0
        vectors,trunc=encode(['query: What country is Ada from?','passage: '+p1.text()],e5tok,e5,a.batch_size)
        assert vectors.shape==(2,384) and not trunc
        ev,labs=render_labeled_evidence(paths,endpoints)
        inputs=tokenizer(build_labeled_prompt(tokenizer,'What country is Ada from?',ev),return_tensors='pt').to('cuda')
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.eos_token_id,repetition_penalty=1.1,no_repeat_ngram_size=3)
        atomic_json(a.output,{'status':'PASS','kind':'synthetic smoke only','budget_checks':checks,
            'response':tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True),'elapsed':time.perf_counter()-start})
        print('SMOKE_PASS',flush=True);return
    official_path=project/'external/webqsp_official/WebQSP/data/WebQSP.test.json'
    audit_path=project/'results/week2_webqsp_resource_audit.json'
    legacy_path=project/'results/week2_webqsp_qwen_test500_B256.json'
    official=json.loads(official_path.read_text());audit=json.loads(audit_path.read_text());legacy=json.loads(legacy_path.read_text())
    assert audit['status']=='PASS' and all(audit['gate'].values())
    auditrows={r['question_id']:r for r in audit['records']};legacy_ids=set(legacy['configuration']['sample_ids'])
    rows=[]
    for q in official['Questions']:
        r=parse_question(q)
        if r and auditrows.get(r['id'],{}).get('subgraph_exists') and auditrows[r['id']].get('topic_present'): rows.append(r)
    rows.sort(key=lambda r:int(r['id'].rsplit('-',1)[-1]));assert len(rows)==1577 and len(legacy_ids)==500
    assert len({r['id'] for r in rows}-legacy_ids)==1077
    vendor_dir=Path(__file__).resolve().parents[1]/'vendor'
    model_dirs=[project/'models/Qwen3-8B-AWQ',project/'revision_20260909/models/e5-small-v2']
    model_hashes={str(p.relative_to(project)):sha(p) for md in model_dirs for p in sorted(md.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.safetensors')}
    config={'vendor_sha256':{p.name:sha(p) for p in sorted(vendor_dir.glob('*.py'))},
        'model_files_sha256':model_hashes,'batch_size':a.batch_size,
        'packages':{p:package_version(p) for p in ['torch','transformers','tokenizers','autoawq','numpy']},
        'conditions':CONDITIONS,'budget':256,'pool_cap':1000,'initial_cap':100000,'depths':[1,2],
        'pool':'top500 full-path BM25 union top500 relation BM25; fill from full-path BM25 to 1000',
        'budget_definition':'full labeled evidence body, exact check at each acceptance','max_new_tokens':32,
        'repetition_penalty':1.1,'no_repeat_ngram_size':3,'do_sample':False,'thinking':False,'seed':20260909,
        'e5_model':'intfloat/e5-small-v2','e5_revision':E5_REV,'e5_dtype':'float32','e5_max_length':512,
        'qwen_model':'Qwen/Qwen3-8B-AWQ','qwen_revision':QWEN_REV,'qwen_compute_dtype':'float16',
        'protocol_sha256':sha(a.protocol),'source_sha256':sha(__file__),'official_sha256':sha(official_path),
        'audit_sha256':sha(audit_path),'legacy_sha256':sha(legacy_path),'ids':[r['id'] for r in rows]}
    if a.resume:
        payload=json.loads(a.output.read_text());assert payload['configuration']==config
        done={(r['id'],r['condition']) for r in payload['results']}
        payload.setdefault('resume_events',[]).append(datetime.now(timezone.utc).isoformat())
    else:
        payload={'status':'RUNNING','created_utc':datetime.now(timezone.utc).isoformat(),'configuration':config,
            'load_seconds':load_seconds,'results':[],'questions':[]};done=set()
    atomic_json(a.output,payload)
    graphdir=project/'external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods'
    for qi,row in enumerate(rows):
        if all((row['id'],c) in done for c in CONDITIONS):continue
        qstart=time.perf_counter();adj,names,triples=load_question_graph(graphdir/(row['id']+'.nxhd'))
        paths,endpoints,truncated=enumerate_paths(adj,names,row['topics'],100000)
        full=bm25_scores(row['question'],paths);rel=relation_bm25_scores(row['question'],paths)
        fullorder=rank_by_scores(paths,full);relorder=rank_by_scores(paths,rel)
        pool=list(dict.fromkeys(fullorder[:500]+relorder[:500]))
        for p in fullorder:
            if len(pool)>=1000:break
            if p not in pool:pool.append(p)
        costs=[count(tokenizer,render_labeled_evidence([p],endpoints)[0]) for p in pool]
        signatures=[p.directional_signature() for p in pool]
        fullmap=dict(zip(paths,full));relmap=dict(zip(paths,rel));fs=[fullmap[p] for p in pool];rs=[relmap[p] for p in pool]
        shared_seconds=time.perf_counter()-qstart
        torch.cuda.synchronize();es=time.perf_counter()
        vectors,trunc=encode(['query: '+row['question']]+['passage: '+p.text() for p in pool],e5tok,e5,a.batch_size)
        dense=(vectors[0:1]@vectors[1:].T).squeeze(0).tolist() if pool else []
        torch.cuda.synchronize();encode_seconds=time.perf_counter()-es
        gold=set(row['gold_ids']);recall=lambda ps: len({endpoints[p] for p in ps}&gold)/len(gold) if gold else 0.
        qmeta={'id':row['id'],'group':'legacy500' if row['id'] in legacy_ids else 'extension1077',
            'annotated_max_hops':row['annotated_max_hops'],'candidate_paths':len(paths),'candidate_truncated':truncated,
            'common_pool_paths':len(pool),'candidate_recall':recall(paths),'common_pool_recall':recall(pool),
            'bfs20k_recall':recall(paths[:20000]),'bm25_20k_recall':recall(fullorder[:20000]),
            'relation20k_recall':recall(relorder[:20000]),'subgraph_triples':triples,
            'negative_relation_scores':sum(s<0 for s in rs),'allzero_relation_scores':not any(s>0 for s in rs),
            'e5_truncated_texts':trunc,'shared_preparation_seconds':shared_seconds,'e5_encoding_seconds':encode_seconds,
            'pool_sha256':hashlib.sha256(json.dumps([p.text() for p in pool],ensure_ascii=False).encode()).hexdigest()}
        if not any(q['id']==row['id'] for q in payload['questions']):payload['questions'].append(qmeta)
        order=CONDITIONS[qi%5:]+CONDITIONS[:qi%5]
        for ci,c in enumerate(order):
            if (row['id'],c) in done:continue
            scores=dense if c.startswith('e5') else (fs if c=='bm25' else rs)
            ss=time.perf_counter();selected,indices=select(pool,scores,endpoints,tokenizer,256,c,costs,signatures)
            ev,labels=render_labeled_evidence(selected,endpoints);tokcount=count(tokenizer,ev);assert tokcount<=256
            selection_seconds=time.perf_counter()-ss
            inputs=tokenizer(build_labeled_prompt(tokenizer,row['question'],ev),return_tensors='pt').to('cuda')
            ninput=inputs['input_ids'].shape[1];torch.cuda.synchronize();gs=time.perf_counter()
            with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=32,do_sample=False,
                use_cache=True,pad_token_id=tokenizer.eos_token_id,repetition_penalty=1.1,no_repeat_ngram_size=3)
            torch.cuda.synchronize();generation_seconds=time.perf_counter()-gs
            output=generated[0,ninput:];response=tokenizer.decode(output,skip_special_tokens=True).strip()
            predicted=strict_codes(response,labels);endpointlist=list(dict.fromkeys(endpoints[p] for p in selected))
            record={**qmeta,'condition':c,'condition_execution_index':ci,'question':row['question'],'topics':row['topics'],
                'gold_ids':row['gold_ids'],'selected_pool_indices':indices,'selected_paths':len(selected),
                'selected_evidence':[p.text() for p in selected],'selected_endpoint_ids':endpointlist,
                'rendered_labeled_evidence':ev,'answer_code_map':labels,'evidence_tokens':tokcount,
                'total_input_tokens':ninput,'output_tokens':len(output),'output_hit_max_new_tokens':len(output)==32,
                'raw_response':response,'mapped_prediction_ids':predicted,**id_metrics(predicted,row['gold_ids']),
                'endpoint_metrics':id_metrics(endpointlist,row['gold_ids']),'evidence_answer_recall':recall(selected),
                'selection_seconds':selection_seconds,'generation_seconds':generation_seconds,
                'method_retrieval_seconds':shared_seconds+selection_seconds+(encode_seconds if c.startswith('e5') else 0.)}
            payload['results'].append(record);done.add((row['id'],c))
        payload['elapsed_this_session_seconds']=time.perf_counter()-start
        atomic_json(a.output,payload)
        print(f"QUESTION {qi+1}/1577 {row['id']} rows={len(payload['results'])} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    assert len(payload['results'])==7885 and len(done)==7885
    payload['status']='PASS';payload['completed_utc']=datetime.now(timezone.utc).isoformat();atomic_json(a.output,payload)
    print('FORMAL_PASS',sha(a.output),flush=True)
if __name__=='__main__':main()
