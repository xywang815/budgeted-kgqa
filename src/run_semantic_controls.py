#!/usr/bin/env python3
"""Frozen E5 comparison and training-selected calibration with exact token caps."""
from __future__ import annotations
import argparse,gzip,hashlib,json,sys,time,random
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
CHOICES=('rank','max_density_g05','max_density_g1','minmax_density_g05','minmax_density_g1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jdump(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)


def scaled_scores(scores,rule):
    if rule=='rank':return list(scores),0.
    minimum=min(scores,default=0.) if rule.startswith('minmax') else 0.
    maximum=max(scores,default=0.);den=maximum-minimum
    rel=[max(0.,s-minimum)/den if den>0 else 0. for s in scores]
    return rel,.5 if rule.endswith('g05') else 1.


def render(paths,indices,labeled):
    return '\n'.join(f'- {paths[i]}' + (f' => A{j+1}' if labeled else '') for j,i in enumerate(indices))


def select_indices(paths,costs,hops,scores,rule,tokenizer,labeled,budget=256):
    values,gamma=scaled_scores(scores,rule);chosen=[]
    def count(indices):
        text=render(paths,indices,labeled)
        return len(tokenizer.encode(text,add_special_tokens=False)) if text else 0
    if not labeled:
        # Match the historical MetaQA rank + two-token standalone screen.
        assert rule=='rank'
        total=0
        for i in sorted(range(len(paths)),key=lambda i:(-values[i],hops[i],i)):
            if costs[i]>(budget-total)+2:continue
            n=count(chosen+[i])
            if n<=budget:chosen.append(i);total=n
        return chosen
    if rule=='rank':order=sorted(range(len(paths)),key=lambda i:(-values[i],i))
    else:order=sorted(range(len(paths)),key=lambda i:(-values[i]/costs[i]**gamma,-values[i],costs[i],i))
    remaining=list(order)
    while remaining:
        accepted=None
        for i in remaining:
            if count(chosen+[i])<=budget:accepted=i;break
        if accepted is None:break
        chosen.append(accepted);remaining.remove(accepted)
    return chosen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--encoder-batch-size',type=int,default=64);ap.add_argument('--reader-batch-size',type=int,default=8)
    ap.add_argument('--timeout-hours',type=float,default=8.)
    a=ap.parse_args();p=a.project.resolve();out=a.output.resolve();sys.path.insert(0,str(p/'src'))
    import fcntl
    lock=(p.parent/'codex-single-gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np,torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
    from run_kgqa_inference import build_prompt,normalize_entity,parse_entities,answer_metrics,running_gpu_compute_processes,package_version
    from run_webqsp_replication import build_labeled_prompt,id_metrics
    from run_reader_interventions import trim_generated_ids,strict_codes
    if running_gpu_compute_processes():raise RuntimeError('GPU occupied')
    assert json.loads((a.candidates/'progress.json').read_text())['status']=='PASS'
    if out.exists() and not a.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);(out/'arrays').mkdir(exist_ok=True)
    index=json.loads((a.candidates/'index.json').read_text());assert len(index)==2077
    for item in index:assert sha(a.candidates/item['file'])==item['sha256']
    original=json.loads((p/'revision_20260909/results/shared_pool_webqsp_all1577_B256.json').read_text())
    for f,h in original['configuration']['model_files_sha256'].items():assert sha(p/f)==h
    config={'source_sha256':sha(__file__),'candidate_index_sha256':sha(a.candidates/'index.json'),
        'candidates_configuration':json.loads((a.candidates/'configuration.json').read_text()),
        'choices':CHOICES,'development_selection':'endpoint macro F1; ties by CHOICES order; frozen before WebQSP test',
        'encoder_batch_size':a.encoder_batch_size,'reader_batch_size':a.reader_batch_size,'seed':20261110,
        'reader_controls':'MetaQA matched original and no-repetition-constraints for BM25/RACPS/E5-rank;96 tokens. WebQSP original constraints,32 tokens.',
        'model_files_sha256':original['configuration']['model_files_sha256'],
        'packages':{n:package_version(n) for n in ['torch','transformers','numpy','autoawq']},
        'helper_sha256':{n:sha(p/'src'/n) for n in ('kgqa_core.py','run_kgqa_inference.py','run_webqsp_replication.py','run_reader_interventions.py')},
        'scope':'post-review exploratory; no test-dependent choice'}
    if a.resume:assert json.loads((out/'configuration.json').read_text())==json.loads(json.dumps(config))
    else:jdump(out/'configuration.json',config)
    torch.set_num_threads(8);torch.manual_seed(20261110);torch.cuda.manual_seed_all(20261110);random.seed(20261110)
    qtok=AutoTokenizer.from_pretrained(p/'models/Qwen3-8B-AWQ',local_files_only=True)
    etok=AutoTokenizer.from_pretrained(p/'revision_20260909/models/e5-small-v2',local_files_only=True)
    encoder=AutoModel.from_pretrained(p/'revision_20260909/models/e5-small-v2',local_files_only=True,torch_dtype=torch.float32).to('cuda').eval()
    def encode(texts):
        chunks=[];truncated=0
        for start in range(0,len(texts),a.encoder_batch_size):
            batch=texts[start:start+a.encoder_batch_size]
            lengths=etok(batch,truncation=False,add_special_tokens=True,return_length=True)['length'];truncated+=sum(n>512 for n in lengths)
            tok=etok(batch,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                hidden=encoder(**tok).last_hidden_state
                hidden=hidden.masked_fill(~tok['attention_mask'][...,None].bool(),0.)
                pooled=hidden.sum(dim=1)/tok['attention_mask'].sum(dim=1)[...,None]
                chunks.append(torch.nn.functional.normalize(pooled,p=2,dim=1))
        return torch.cat(chunks),truncated
    started=time.monotonic();records={};selection_path=out/'selection_records.jsonl'
    if a.resume and selection_path.exists():
        records={r['phase']+':'+r['id']:r for r in map(json.loads,selection_path.read_text().splitlines())}
    historical={(r['id'],r['condition']):r for r in original['results']}
    dev_scores=defaultdict(list);chosen=None
    def pick():
        means={c:float(np.mean(dev_scores[c])) for c in CHOICES};assert all(len(dev_scores[c])==100 for c in CHOICES)
        winner=max(CHOICES,key=lambda c:(means[c],-CHOICES.index(c)))
        receipt={'rule':winner,'development_endpoint_macro_f1':means,'development_n':100,'created_before_test_scoring':True,
            'criterion':'macro endpoint F1, CHOICES order for exact ties','test_results_used':False}
        path=out/'development_choice.json'
        if path.exists():assert json.loads(path.read_text())==receipt
        else:jdump(path,receipt)
        return winner
    with selection_path.open('a',buffering=1) as h:
      for item in index:
        if time.monotonic()-started>a.timeout_hours*3600:raise TimeoutError('semantic scoring timeout; records preserved')
        key=item['phase']+':'+item['id']
        if item['phase']!='web_dev' and chosen is None:chosen=pick()
        if key in records:
            if item['phase']=='web_dev':
                for rule,v in records[key]['selections'].items():dev_scores[rule].append(v['endpoint_metrics']['f1'])
            continue
        with gzip.open(a.candidates/item['file'],'rt') as f:cache=json.load(f)
        torch.cuda.synchronize();t=time.perf_counter()
        vectors,truncated=encode(['query: '+cache['question']]+['passage: '+x for x in cache['paths']])
        scores=(vectors[:1]@vectors[1:].T).squeeze(0).tolist();del vectors;torch.cuda.synchronize();encoding_seconds=time.perf_counter()-t
        labeled=item['phase']!='meta_test';rules=CHOICES if item['phase']=='web_dev' else tuple(dict.fromkeys(['rank','max_density_g1',chosen])) if labeled else ('rank',)
        selections={}
        for rule in rules:
            ids=select_indices(cache['paths'],cache['costs'],cache['hops'],scores,rule,qtok,labeled)
            evidence=render(cache['paths'],ids,labeled);tokens=len(qtok.encode(evidence,add_special_tokens=False)) if evidence else 0;assert tokens<=256
            eps=[cache['endpoints'][i] for i in ids]
            if not labeled:eps=[normalize_entity(x) for x in eps]
            scoring=id_metrics if labeled else answer_metrics
            selections[rule]={'indices':ids,'paths':[cache['paths'][i] for i in ids],'endpoints':eps,'rendered_evidence':evidence,
                'evidence_tokens':tokens,'selected_hops':[cache['hops'][i] for i in ids],'endpoint_metrics':scoring(eps,cache['gold_answers'])}
            if item['phase']=='web_dev':dev_scores[rule].append(selections[rule]['endpoint_metrics']['f1'])
        array_name=f'{item["phase"]}_{item["id"]}.npz'
        np.savez_compressed(out/'arrays'/array_name,cosine=np.asarray(scores,dtype=np.float32),cost=np.asarray(cache['costs'],dtype=np.int32),hops=np.asarray(cache['hops'],dtype=np.uint8))
        row={k:cache[k] for k in ('id','phase','question','gold_answers','stratum','pool_sha256','candidate_paths','candidate_truncated')}
        if item['phase']=='web_test':
            row['historical_selection_agreement']={rule:selections[rule]['indices']==historical[(row['id'],method)]['selected_pool_indices'] for rule,method in [('rank','e5_rank'),('max_density_g1','e5_density')]}
        row.update(pool_paths=len(cache['paths']),selections=selections,arrays=array_name,arrays_sha256=sha(out/'arrays'/array_name),
            encoding_seconds=encoding_seconds,encoder_truncated_texts=truncated,cosine_quantiles=np.quantile(scores,[0,.1,.5,.9,1]).tolist() if scores else [],
            cost_quantiles=np.quantile(cache['costs'],[0,.1,.5,.9,1]).tolist() if cache['costs'] else [])
        h.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');records[key]=row
        if len(records)%5==0:
            jdump(out/'progress.json',{'status':'SCORING','completed_pools':len(records),'expected_pools':len(index),'elapsed_seconds':time.monotonic()-started,'last_id':key})
            print('SCORE',len(records),'/',len(index),key,flush=True)
    chosen=chosen or pick();del encoder;import gc;gc.collect();torch.cuda.empty_cache()
    jobs=[]
    for row in records.values():
        if row['phase']=='web_dev':continue
        for label,rule in ([('e5_rank','rank')] if row['phase']=='meta_test' else [('e5_rank','rank'),('e5_density','max_density_g1'),('e5_development_choice',chosen)]):
            selected=row['selections'][rule]
            jobs.append({'dataset':'MetaQA' if row['phase']=='meta_test' else 'WebQSP','id':row['id'],'stratum':row['stratum'],
                'selector':label,'rule':rule,'question':row['question'],'gold_answers':row['gold_answers'],**selected})
    meta=json.loads((p/'results/qwen3_awq_test300_B256.json').read_text())
    for r in meta['results']:
        if r['condition'] not in ('bm25','ra_mmr'):continue
        jobs.append({'dataset':'MetaQA','id':r['id'],'stratum':r['hop'],'selector':'racps' if r['condition']=='ra_mmr' else 'bm25',
            'rule':'original selected paths','question':r['question'],'gold_answers':r['gold_answers'],'paths':r['selected_evidence'],
            'endpoints':[normalize_entity(x) for x in r['selected_endpoints']],'rendered_evidence':'\n'.join('- '+x for x in r['selected_evidence']),
            'evidence_tokens':r['evidence_tokens'],'historical_f1':r['f1'],'historical_raw_response':r['raw_response']})
    assert len(jobs)==5931
    jobs=[{**job,'condition':condition} for job in jobs for condition in
          (('original','no_rep_constraints') if job['dataset']=='MetaQA' else ('original',))]
    assert len(jobs)==8631
    model=AutoModelForCausalLM.from_pretrained(p/'models/Qwen3-8B-AWQ',local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,
        device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval();qtok.padding_side='left';qtok.pad_token=qtok.eos_token
    done=set();output_path=out/'reader_records.jsonl'
    if a.resume and output_path.exists():done={r['job_id'] for r in map(json.loads,output_path.read_text().splitlines())}
    pending=defaultdict(list);prompts={}
    for job in jobs:
        job['job_id']=job['dataset']+':'+job['selector']+':'+job['condition']+':'+job['id']
        if job['job_id'] in done:continue
        webmode=job['dataset']=='WebQSP';cap=32 if webmode else 96
        prompt=build_labeled_prompt(qtok,job['question'],job['rendered_evidence']) if webmode else build_prompt(qtok,job['question'],job['rendered_evidence'])[0]
        assert len(qtok.encode(job['rendered_evidence'],add_special_tokens=False))==job['evidence_tokens']
        penalty=1.0 if job['condition']=='no_rep_constraints' else 1.1
        ngram=0 if job['condition']=='no_rep_constraints' else 3
        job['prompt']=prompt;job['total_input_tokens']=len(qtok(prompt)['input_ids']);job['max_new_tokens']=cap
        job['generation_configuration']={'max_new_tokens':cap,'repetition_penalty':penalty,'no_repeat_ngram_size':ngram,'do_sample':False,'use_cache':True}
        key=hashlib.sha256((json.dumps(job['generation_configuration'],sort_keys=True)+'|'+prompt).encode()).hexdigest();job['cache_key']=key;pending[key].append(job);prompts[key]=prompt
    buckets=defaultdict(list)
    for key,jj in pending.items():buckets[json.dumps(jj[0]['generation_configuration'],sort_keys=True)].append(key)
    for keys in buckets.values():keys.sort(key=lambda k:(pending[k][0]['total_input_tokens'],k))
    with output_path.open('a',buffering=1) as h:
      for genconf,keys in buckets.items():
        genconf=json.loads(genconf);cap=genconf['max_new_tokens']
        for start in range(0,len(keys),a.reader_batch_size):
            if time.monotonic()-started>a.timeout_hours*3600:raise TimeoutError('semantic reader timeout')
            batch=keys[start:start+a.reader_batch_size];inp=qtok([prompts[k] for k in batch],padding=True,return_tensors='pt').to('cuda')
            width=inp['input_ids'].shape[1];torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():outids=model.generate(**inp,**genconf,pad_token_id=qtok.eos_token_id)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            for j,key in enumerate(batch):
                ids=trim_generated_ids(outids[j,width:].tolist(),model.generation_config.eos_token_id);raw=qtok.decode(ids,skip_special_tokens=True).strip()
                for job in pending[key]:
                    webmode=job['dataset']=='WebQSP';labels={f'a{i+1}':x for i,x in enumerate(job['endpoints'])}
                    pred=strict_codes(raw,labels) if webmode else parse_entities(raw);score=id_metrics if webmode else answer_metrics
                    row={**job,'raw_response':raw,'output_ids':ids,'output_tokens':len(ids),'prediction':pred,'metrics':score(pred,job['gold_answers']),
                        'endpoint_metrics':score(job['endpoints'],job['gold_answers']),'hit_generation_cap':len(ids)>=cap,
                        'batch_wall_seconds':seconds,'actual_batch_size':len(batch),'timing_scope':'amortized batch generation; not end-to-end latency',
                        'reused_job_id':None if job is pending[key][0] else pending[key][0]['job_id']}
                    h.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');done.add(job['job_id'])
            jdump(out/'progress.json',{'status':'READING','completed_pools':len(index),'completed_reader_rows':len(done),'expected_reader_rows':len(jobs),'elapsed_seconds':time.monotonic()-started})
            if start%(a.reader_batch_size*10)==0:print('READ',len(done),'/',len(jobs),flush=True)
    jdump(out/'progress.json',{'status':'PASS','completed_pools':len(index),'completed_reader_rows':len(done),'expected_reader_rows':len(jobs),'elapsed_seconds':time.monotonic()-started})
    print('PASS',len(jobs),flush=True)

if __name__=='__main__':main()
