#!/usr/bin/env python3
"""Frozen cost-by-decoder confirmation on previously unused MetaQA questions.

No hyperparameter search. All four cells and all sampled questions are retained.
The sample and input hashes are written before selection or model generation.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time, os
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def qkey(q): return ' '.join(q.casefold().replace('[','').replace(']','').split())
def walk(x):
    if isinstance(x,dict):
        if str(x.get('id','')).startswith('metaqa-') and 'question' in x: yield x
        else:
            for v in x.values(): yield from walk(v)
    elif isinstance(x,list):
        for v in x: yield from walk(v)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();p=a.project.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());assert plan['n_per_hop']==100 and plan['seed']==20261111
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from kgqa_core import load_kg,load_jsonl,enumerate_simple_paths,standalone_evidence_tokens,relation_bm25_scores,budget_aware_mmr,render_evidence
    from run_kgqa_inference import build_prompt,normalize_entity,parse_entities,answer_metrics,running_gpu_compute_processes,package_version
    from run_reader_interventions import trim_generated_ids,generation_config
    torch.set_num_threads(8)
    random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    modeldir=p/'models/Qwen3-8B-AWQ';tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    data=p/'data/processed/metaqa'
    manifest_path=out/'sample_manifest.json'
    if not manifest_path.exists():
        seen_ids=set();seen_q=set();source_hashes={}
        files=sorted(set((p/'results').glob('*.json'))|set(p.glob('revision*/results/*.json')))
        for f in files:
            obj=json.loads(f.read_text());rr=list(walk(obj))
            if rr:
                source_hashes[str(f.relative_to(p))]=sha(f)
                seen_ids.update(r['id'] for r in rr);seen_q.update(qkey(r['question']) for r in rr)
        sampled=[];eligible={}
        for hop in (1,2,3):
            rows=load_jsonl(data/f'{hop}-hop/test.jsonl')
            rows=[r for r in rows if r['id'] not in seen_ids and qkey(r['question']) not in seen_q]
            rows.sort(key=lambda r:hashlib.sha256(f"{plan['seed']}:{r['id']}".encode()).hexdigest())
            unique=[];qq=set()
            for r in rows:
                if qkey(r['question']) not in qq:unique.append(r);qq.add(qkey(r['question']))
            eligible[str(hop)]=len(unique);sampled.extend(unique[:plan['n_per_hop']])
        assert len(sampled)==300 and len({r['id'] for r in sampled})==300
        dump(manifest_path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),
            'selection_rule':'lowest SHA256(seed:question_id), 100 unique normalized questions per hop',
            'excluded_ids':sorted(seen_ids),'excluded_normalized_questions':len(seen_q),'eligible_unique_by_hop':eligible,
            'exclusion_source_hashes':source_hashes,'processed_data_hashes':{str(f.relative_to(data)):sha(f) for f in [data/'kg.tsv',*[data/f'{h}-hop/test.jsonl' for h in (1,2,3)]]},
            'sample':sampled,'overlap_ids':0,'overlap_normalized_questions':0})
    manifest=json.loads(manifest_path.read_text());assert manifest['plan_sha256']==sha(a.plan)
    sampled=manifest['sample'];selection_path=out/'selections.jsonl'
    completed={}
    if selection_path.exists():
        for line in selection_path.read_text().splitlines():
            r=json.loads(line);completed[r['id']]=r
    if len(completed)<300:
        adj=load_kg(data/'kg.tsv')
        with selection_path.open('a',buffering=1) as handle:
            for i,r in enumerate(sampled):
                if r['id'] in completed:continue
                t=time.monotonic();paths,hit=enumerate_simple_paths(adj,r['topic_entity'],r['hop'],100000,exact_hops_only=True)
                costs=standalone_evidence_tokens(paths,tok);scores=relation_bm25_scores(r['question'],paths)
                signatures=[v.directional_signature() for v in paths];sel={}
                for name,gamma in [('full',1.),('no_cost',0.)]:
                    kept=budget_aware_mmr(paths,scores,256,tok,relevance_weight=.9,cost_exponent=gamma,standalone_costs=costs,signatures=signatures)
                    evidence=render_evidence(kept);prompt=build_prompt(tok,r['question'],evidence)[0]
                    nt=len(tok.encode(evidence,add_special_tokens=False)) if evidence else 0;assert nt<=256
                    sel[name]={'selected_evidence':[v.text() for v in kept],'selected_endpoints':list(dict.fromkeys(normalize_entity(v.nodes[-1]) for v in kept)),
                        'rendered_evidence':evidence,'evidence_tokens':nt,'prompt':prompt,'total_input_tokens':len(tok(prompt)['input_ids'])}
                result={**r,'candidate_paths':len(paths),'candidate_cap_hit':hit,'selection_seconds':time.monotonic()-t,'selections':sel}
                handle.write(json.dumps(result,ensure_ascii=False)+'\n');completed[r['id']]=result
                dump(out/'progress.json',{'status':'PREPARING','selected_questions':len(completed),'expected_questions':300})
                if len(completed)%10==0:print('SELECTED',len(completed),flush=True)
        del adj
    if a.prepare_only:return
    import fcntl
    lock=(p.parent/'codex-single-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if running_gpu_compute_processes():raise RuntimeError('GPU occupied; no process stopped')
    conf={'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'manifest_sha256':sha(manifest_path),
        'selection_sha256':sha(selection_path),'script_sha256':sha(__file__),'helpers':{n:sha(p/'src'/n) for n in ['kgqa_core.py','run_kgqa_inference.py','run_reader_interventions.py']},
        'model_sha256':{f.name:sha(f) for f in sorted(modeldir.iterdir()) if f.is_file()},
        'environment':{n:package_version(n) for n in ['torch','transformers','tokenizers','autoawq','numpy']},
        'batch_size':8,'generation_cap':96,'n_questions':300,'n_conditions':4,'sample_scope':'new questions, same model and benchmark, templates may overlap'}
    if (out/'configuration.json').exists():
        old=json.loads((out/'configuration.json').read_text());conf['created_utc']=old['created_utc'];assert conf==old
    else:dump(out/'configuration.json',conf)
    record_path=out/'records.jsonl';done=set();cache={}
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            r=json.loads(line);done.add(r['job_id']);cache[r['cache_key']]=r
    jobs={};buckets=defaultdict(list)
    for r0 in sampled:
        r=completed[r0['id']]
        for selector in ('full','no_cost'):
            for decoder in ('original','no_rep_constraints'):
                jid=f"{r['id']}:{selector}:{decoder}"
                if jid in done:continue
                x=r['selections'][selector];g=generation_config(decoder,'MetaQA');ids=tok(x['prompt'])['input_ids']
                key=hashlib.sha256(json.dumps([ids,g],sort_keys=True).encode()).hexdigest()
                item={**x,'id':r['id'],'job_id':jid,'question':r['question'],'topic_entity':r['topic_entity'],'stratum':r['hop'],
                    'gold_answers':r['gold_answers'],'selector':selector,'condition':decoder,'generation_configuration':g,'cache_key':key}
                jobs.setdefault(key,[]).append(item)
    for key,items in jobs.items():buckets[json.dumps(items[0]['generation_configuration'],sort_keys=True)].append(key)
    model=AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    start=time.monotonic();generated=0;batchno=0
    with record_path.open('a',buffering=1) as handle:
        def save(key,base):
            for item in jobs[key]:
                pred=parse_entities(base['raw_response']);gold=item['gold_answers'];ends=set(item['selected_endpoints'])
                row={**item,**{k:base[k] for k in ['raw_response','output_ids','batch_number','padded_input_width']},
                    'output_tokens':len(base['output_ids']),'hit_generation_cap':len(base['output_ids'])>=96,
                    'reused_job_id':None if item['job_id']==base['job_id'] else base['job_id'],
                    'prediction':pred,'metrics':answer_metrics(pred,gold),'filtered_metrics':answer_metrics([v for v in pred if v in ends],gold),
                    'endpoint_metrics':answer_metrics(list(ends),gold)}
                handle.write(json.dumps(row,ensure_ascii=False)+'\n');done.add(item['job_id'])
        for cf,keys in sorted(buckets.items()):
            for k in [k for k in keys if k in cache]:save(k,cache[k])
            keys=[k for k in keys if k not in cache];keys.sort(key=lambda k:(jobs[k][0]['total_input_tokens'],k))
            for offset in range(0,len(keys),8):
                if time.monotonic()-start>10800:raise TimeoutError('Three hour inference limit, resumable')
                batch=keys[offset:offset+8];inputs=tok([jobs[k][0]['prompt'] for k in batch],padding=True,return_tensors='pt').to('cuda');width=inputs['input_ids'].shape[1]
                with torch.inference_mode():outputs=model.generate(**inputs,**json.loads(cf),pad_token_id=tok.eos_token_id)
                batchno+=1;generated+=len(batch)
                for j,k in enumerate(batch):
                    ids=trim_generated_ids(outputs[j,width:].tolist(),model.generation_config.eos_token_id)
                    base={'job_id':jobs[k][0]['job_id'],'output_ids':ids,'raw_response':tok.decode(ids,skip_special_tokens=True).strip(),'batch_number':batchno,'padded_input_width':width};save(k,base);cache[k]=base
                dump(out/'progress.json',{'status':'RUNNING','completed':len(done),'expected':1200,'generated_this_session':generated,'elapsed_seconds':time.monotonic()-start})
                if batchno%5==0:print('GENERATED',len(done),'/1200',flush=True)
    assert len(done)==1200
    dump(out/'progress.json',{'status':'PASS','completed':1200,'expected':1200,'generated_this_session':generated,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
