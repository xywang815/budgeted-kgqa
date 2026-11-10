#!/usr/bin/env python3
"""Matched reader interventions on immutable saved path selections (no retrieval).

Matched greedy baselines are rerun with the same batch configuration as interventions.
Every condition is retained, including context-over-cap order/interface probes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

META_CONDITIONS = ('original', 'no_cost_original', 'labeled_names', 'labeled_codes',
                   'canonical', 'no_cost_canonical', 'cap192', 'no_rep_penalty',
                   'no_ngram_block', 'no_rep_constraints', 'labeled_names_no_constraints', 'labeled_codes_no_constraints')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal(path):
    return re.split(r'(?:--\[[^\]]+\]-->|<--\[[^\]]+\]--)\s*', path)[-1]


def ordered_paths(paths, condition):
    result = sorted(paths) if 'canonical' in condition else list(paths)
    assert Counter(result) == Counter(paths)
    return result


def strict_codes(response, labels):
    return list(dict.fromkeys(labels.get(c.lower(), '__invalid_code__' + c.lower())
                for c in re.findall(r'(?i)\bA\d+\b', response)))


def generation_config(condition, dataset):
    cap = 32 if dataset == 'WebQSP' else 96
    if condition == 'cap64': cap = 64
    if condition == 'cap192': cap = 192
    return {'max_new_tokens': cap, 'do_sample': False, 'use_cache': True,
            'repetition_penalty': 1.0 if (condition in ('no_rep_penalty', 'no_rep_constraints') or condition.endswith('_no_constraints')) else 1.1,
            'no_repeat_ngram_size': 0 if (condition in ('no_ngram_block', 'no_rep_constraints') or condition.endswith('_no_constraints')) else 3}


def matched_interface_prompt(tokenizer, question, evidence, codes):
    common = ('Answer the knowledge-graph question using only the candidate paths. '
              'Each evidence line is a candidate reasoning path followed by an answer code. '
              'The entity immediately before the => marker is the candidate answer; '
              'the code after the marker identifies that path. ')
    if codes:
        output = 'Return the code or codes whose candidate entities answer the question. '
    else:
        output = 'Return the candidate entity name or names that answer the question, preserving their spelling. '
    system = common + output + ('Separate multiple answers with |. Return each answer at most once. '
                                'If no candidate is valid, return NONE. Return answers only, with no explanation.')
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': f'Evidence:\n{evidence}\n\nQuestion: {question}\n/no_think'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefill = '<think>\n\n</think>\n\n'
    return text if text.endswith(prefill) else text + prefill


def trim_generated_ids(ids, eos_ids):
    ends = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])
    for i, token in enumerate(ids):
        if token in ends: return ids[:i + 1]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--group', choices=['meta256', 'meta-other-caps', 'web-cap', 'all'], default='all')
    ap.add_argument('--limit-per-stratum', type=int, default=0, help='Smoke only; zero keeps every question.')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--timeout-hours', type=float, default=8.)
    a = ap.parse_args()
    if a.limit_per_stratum < 0 or a.timeout_hours <= 0 or a.batch_size < 1: raise ValueError('Invalid limit or timeout')
    p = a.project.resolve(); out = a.output.resolve()
    sys.path.insert(0, str(p / 'src'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_kgqa_inference import (atomic_json, build_prompt, normalize_entity, parse_entities,
                                   answer_metrics, running_gpu_compute_processes, package_version)
    from run_webqsp_replication import build_labeled_prompt, id_metrics
    import fcntl
    gpu_lock = (p.parent / 'codex-single-gpu.lock').open('a+')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if running_gpu_compute_processes(): raise RuntimeError('GPU occupied; no process will be stopped.')
    if out.exists() and not a.resume: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    def load(path):
        sources[str(path.relative_to(p))] = sha(path)
        x = json.loads(path.read_text()); assert x['status'] == 'PASS'
        return x
    meta = {b:load(p/f'results/qwen3_awq_test300_B{b}.json') for b in (128,256,512)}
    comp = load(p/'results/clean_component_ablation_qwen_B256.json')
    web = load(p/'revision_20260909/results/shared_pool_webqsp_all1577_B256.json')
    for name, digest in web['configuration']['model_files_sha256'].items():
        if name.startswith('models/Qwen3-8B-AWQ/'):
            assert sha(p/name) == digest, name
    configuration = {'group':a.group,'limit_per_stratum':a.limit_per_stratum,
        'seed':20261110, 'batch_size':a.batch_size, 'torch_num_threads':8,
        'batching':'same generation config; length-sorted prompts; left padding; trim at first generated EOS',
        'meta_conditions':META_CONDITIONS,'web_selectors':['racps','e5_rank'],
        'timeout_hours':a.timeout_hours,'sources_sha256':sources,'script_sha256':sha(__file__),
        'helper_sha256':{n:sha(p/'src'/n) for n in ('run_kgqa_inference.py','run_webqsp_replication.py','kgqa_core.py')},
        'model_id':'Qwen/Qwen3-8B-AWQ','model_revision':'cb7d6a337aadb4d2082ed0dcef1032e4f8645194',
        'model_files_sha256':{n:h for n,h in web['configuration']['model_files_sha256'].items() if n.startswith('models/Qwen3-8B-AWQ/')},
        'environment':{n:package_version(n) for n in ('torch','transformers','tokenizers','autoawq','autoawq-kernels','numpy')},
        'scope':'post-review fixed-path sensitivity; preserve original primary predictions',
        'context_policy':'never drop members; record actual context tokens; compare cap-compliant subset separately',
        'cache_policy':'reuse identical prompt IDs and generation parameters within this run; never reuse old outputs'}
    config_path=out/'configuration.json'
    if a.resume:
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(configuration)), 'Resume fingerprint mismatch'
    else: atomic_json(config_path, configuration)
    random.seed(20261110);torch.manual_seed(20261110);torch.cuda.manual_seed_all(20261110)
    torch.set_num_threads(8);torch.cuda.set_device(0)
    tokenizer=AutoTokenizer.from_pretrained(p/'models/Qwen3-8B-AWQ',local_files_only=True,trust_remote_code=False)
    tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(p/'models/Qwen3-8B-AWQ',local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    warmup=tokenizer(build_prompt(tokenizer,'What is the capital of France?','- France --[capital]--> Paris')[0],return_tensors='pt').to('cuda')
    with torch.inference_mode():model.generate(**warmup,max_new_tokens=2,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize();del warmup
    jobs=[]
    no_cost={r['id']:r for r in comp['results'] if r['condition']=='racps_no_cost'}
    def limited(rows, web_mode=False):
        counts=Counter(); result=[]
        for r in rows:
            st=r['annotated_max_hops'] if web_mode else r['hop']
            if a.limit_per_stratum and counts[st]>=a.limit_per_stratum: continue
            counts[st]+=1;result.append(r)
        return result
    for budget in (256,128,512):
        if (budget==256 and a.group not in ('meta256','all')) or (budget!=256 and a.group not in ('meta-other-caps','all')):continue
        rows=limited([r for r in meta[budget]['results'] if r['condition']=='ra_mmr'])
        for i,r in enumerate(rows):
            conditions=list(META_CONDITIONS) if budget==256 else ['original','cap192']
            conditions=conditions[i%len(conditions):]+conditions[:i%len(conditions)]
            for condition in conditions:
                source=no_cost[r['id']] if condition.startswith('no_cost') else r
                jobs.append(('MetaQA',budget,'racps',condition,source))
    if a.group in ('web-cap','all'):
        for method in ('racps','e5_rank'):
            rows=limited([r for r in web['results'] if r['group']=='extension1077' and r['condition']==method],True)
            for i,r in enumerate(rows):
                for condition in (('original','cap64') if i%2==0 else ('cap64','original')):
                    jobs.append(('WebQSP',256,method,condition,r))
    records_path=out/'records.jsonl';done={};cache={}
    if a.resume and records_path.exists():
        for line in records_path.read_text().splitlines():
            row=json.loads(line);assert row['job_id'] not in done
            done[row['job_id']]=True;cache[row['cache_key']]=row
    started=time.monotonic();generated=0;cached=0
    def state(status,**extra):
        atomic_json(out/'progress.json',{'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'completed':len(done),'expected':len(jobs),'new_generations_this_session':generated,
            'cache_hits_this_session':cached,'elapsed_seconds':time.monotonic()-started,**extra})
    state('PREPARING');print('PREPARING',len(jobs),'jobs',flush=True)
    pending={}
    for dataset,budget,selector,condition,source in jobs:
        jobid=f'{dataset}:{budget}:{selector}:{condition}:{source["id"]}'
        if jobid in done:continue
        paths=ordered_paths(source['selected_evidence'],condition)
        if dataset=='MetaQA':
            endpoints=[normalize_entity(terminal(t)) for t in paths]
            assert set(endpoints)==set(map(normalize_entity,source['selected_endpoints']))
            labels={f'a{i+1}':v for i,v in enumerate(endpoints)}
            evidence='\n'.join('- '+x for x in paths)
            if condition.startswith('labeled_'):
                evidence='\n'.join(f'- {x} => A{i+1}' for i,x in enumerate(paths))
                prompt=matched_interface_prompt(tokenizer,source['question'],evidence,condition.startswith('labeled_codes'))
            else:prompt=build_prompt(tokenizer,source['question'],evidence)[0]
            gold=source['gold_answers'];stratum=source['hop']
        else:
            evidence=source['rendered_labeled_evidence'];labels=source['answer_code_map']
            endpoints=source['selected_endpoint_ids'];gold=source['gold_ids'];stratum=source['annotated_max_hops']
            prompt=build_labeled_prompt(tokenizer,source['question'],evidence)
        evidence_tokens=len(tokenizer.encode(evidence,add_special_tokens=False)) if evidence else 0
        ids=tokenizer(prompt)['input_ids'];n=len(ids)
        if condition in ('original','no_cost_original'):
            assert evidence_tokens==source['evidence_tokens'],jobid
            assert n==source['total_input_tokens'],jobid
        genconf=generation_config(condition,dataset)
        key=hashlib.sha256(json.dumps([ids,genconf],sort_keys=True).encode()).hexdigest()
        item={'job_id':jobid,'dataset':dataset,'id':source['id'],'stratum':stratum,'selection_budget':budget,
            'selector':selector,'condition':condition,'question':source['question'],'gold_answers':gold,
            'selected_evidence':paths,'selected_endpoints':list(dict.fromkeys(endpoints)),
            'rendered_evidence':evidence,'prompt':prompt,'evidence_tokens':evidence_tokens,
            'original_evidence_tokens':source['evidence_tokens'],'context_over_selection_cap':evidence_tokens>budget,
            'total_input_tokens':n,'answer_code_map':labels,'generation_configuration':genconf,
            'historical_raw_response':source['raw_response'],'historical_f1':source['f1'],
            'historical_output_tokens':source['output_tokens'],'cache_key':key}
        pending.setdefault(key,[]).append(item)
    from collections import defaultdict
    buckets=defaultdict(list)
    for key,items in pending.items():
        conf=json.dumps(items[0]['generation_configuration'],sort_keys=True)
        buckets[conf].append(key)
    for keys in buckets.values():keys.sort(key=lambda key:(pending[key][0]['total_input_tokens'],key))
    state('RUNNING');print('START',len(jobs),'rows',len(pending),'unique prompts',flush=True)
    eos_ids=model.generation_config.eos_token_id
    batch_number=0
    with records_path.open('a',buffering=1) as handle:
      def save(key,base):
        nonlocal cached
        for item in pending[key]:
            is_code=item['dataset']=='WebQSP' or item['condition'].startswith('labeled_codes')
            pred=strict_codes(base['raw_response'],item['answer_code_map']) if is_code else parse_entities(base['raw_response'])
            score=id_metrics if item['dataset']=='WebQSP' else answer_metrics
            filtered=[x for x in pred if x in set(item['selected_endpoints'])]
            reuse=None if base['job_id']==item['job_id'] else base['job_id']
            if reuse:cached+=1
            result={**item,'raw_response':base['raw_response'],'output_ids':base['output_ids'],
                'output_tokens':len(base['output_ids']),
                'hit_generation_cap':len(base['output_ids'])>=item['generation_configuration']['max_new_tokens'],
                'prediction':pred,'metrics':score(pred,item['gold_answers']),
                'filtered_metrics':score(filtered,item['gold_answers']),
                'endpoint_metrics':score(item['selected_endpoints'],item['gold_answers']),
                'generation_seconds':base['generation_seconds'] if not reuse else 0.,
                'timing_scope':'batch-wall-time amortized over generated prompts; not request latency',
                'batch_number':base['batch_number'],'batch_wall_seconds':base['batch_wall_seconds'],
                'actual_batch_size':base['actual_batch_size'],'padded_input_width':base['padded_input_width'],
                'reused_job_id':reuse}
            handle.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n');done[item['job_id']]=True
      for conf,keys in buckets.items():
        for key in [k for k in keys if k in cache]:save(key,cache[key])
        keys=[k for k in keys if k not in cache]
        for start in range(0,len(keys),a.batch_size):
            if time.monotonic()-started > a.timeout_hours*3600:
                state('TIMEOUT');raise TimeoutError('Hard timeout; completed records preserved')
            batch=keys[start:start+a.batch_size]
            inputs=tokenizer([pending[k][0]['prompt'] for k in batch],padding=True,return_tensors='pt').to('cuda')
            width=inputs['input_ids'].shape[1];genconf=json.loads(conf)
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():output=model.generate(**inputs,**genconf,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;batch_number+=1;generated+=len(batch)
            for j,key in enumerate(batch):
                ids=trim_generated_ids(output[j,width:].tolist(),eos_ids)
                base={'job_id':pending[key][0]['job_id'],'raw_response':tokenizer.decode(ids,skip_special_tokens=True).strip(),
                    'output_ids':ids,'generation_seconds':seconds/len(batch),'batch_number':batch_number,
                    'batch_wall_seconds':seconds,'actual_batch_size':len(batch),'padded_input_width':width}
                save(key,base);cache[key]=base
            handle.flush();state('RUNNING',last_batch=batch_number)
            if batch_number%5==0:print('PROGRESS',len(done),'/',len(jobs),'elapsed',round(time.monotonic()-started,1),flush=True)
    state('PASS');print('PASS',len(done),flush=True)


if __name__=='__main__':
    main()
