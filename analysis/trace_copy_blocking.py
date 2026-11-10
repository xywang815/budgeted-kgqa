#!/usr/bin/env python3
"""Tokenizer-only audit of saved trajectories; performs no model generation.

Potential exposure uses exact token triples for endpoint spellings in two answer
contexts. Actual-trajectory diagnostics replay the n-gram processor on the saved
unblocked trajectory. A violation is counterfactual for that trajectory; it is
not proof of a latent model preference in a different generation history.
"""
import argparse,json,gzip,re,hashlib
from pathlib import Path
from collections import defaultdict

def loadrows(p):
    with gzip.open(p,'rt') if p.suffix=='.gz' else p.open() as h:
        return [json.loads(x) for x in h]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--records',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer,NoRepeatNGramLogitsProcessor
    import torch
    torch.set_num_threads(2)
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from run_kgqa_inference import normalize_entity
    from run_reader_interventions import terminal
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True);proc=NoRepeatNGramLogitsProcessor(3)
    rows=loadrows(a.records);groups=defaultdict(dict)
    for r in rows:
        if r['dataset']=='MetaQA' and r['selection_budget']==256 and r['condition'] in ('original','no_ngram_block','no_rep_constraints'):
            groups[r['id']][r['condition']]=r
    output=[];cases=[];eligible_count=0;exposed_count=0
    for qid,g in sorted(groups.items()):
        assert len(g)==3
        original=g['original'];relaxed=g['no_ngram_block'];assert original['prompt']==relaxed['prompt']
        prompt=tok(original['prompt'])['input_ids'];triples={tuple(prompt[i:i+3]) for i in range(len(prompt)-2)}
        gold=set(map(normalize_entity,original['gold_answers']));ends=set(original['selected_endpoints']);eligible=gold&ends
        spellings={normalize_entity(terminal(x)):terminal(x) for x in original['selected_evidence']}
        exposed=set()
        for entity in eligible:
            for prefix in ('',' '):
                ids=tok.encode(prefix+spellings[entity],add_special_tokens=False)
                if any(tuple(ids[j:j+3]) in triples for j in range(len(ids)-2)):exposed.add(entity)
        eligible_count+=len(eligible);exposed_count+=len(exposed)
        ids=relaxed['output_ids'];decoded=tok.decode(ids,skip_special_tokens=True)
        spans=[]
        for m in re.finditer(r'[^|;\n]+',decoded):
            entity=normalize_entity(m.group())
            if entity in eligible:spans.append((m.start(),m.end(),entity))
        hist=list(prompt);ng=defaultdict(set)
        for j in range(len(hist)-2):ng[tuple(hist[j:j+2])].add(hist[j+2])
        violations=[];correct_blocked=set();shared=True
        for j,token in enumerate(ids):
            banned=token in ng.get(tuple(hist[-2:]),set())
            if banned:
                before=tok.decode(ids[:j],skip_special_tokens=True);after=tok.decode(ids[:j+1],skip_special_tokens=True)
                monotonic=decoded.startswith(before) and decoded.startswith(after)
                entities=[e for s,t,e in spans if monotonic and len(before)<t and len(after)>s]
                correct_blocked.update(entities)
                # Validate the replay predicate against the installed HF processor.
                score=proc(torch.tensor([hist]),torch.zeros((1,len(tok))))
                assert torch.isneginf(score[0,token])
                tri=hist[-2:]+[token];positions=[k for k in range(len(prompt)-2) if prompt[k:k+3]==tri]
                event={'output_index':j,'token_id':token,'token_text':tok.decode([token]),'trigram_ids':tri,'trigram_tokens':tok.convert_ids_to_tokens(tri),
                    'prompt_trigram_positions':positions,'correct_endpoint_entities':entities,'shared_prefix_with_original':original['output_ids'][:j]==ids[:j],
                    'original_next_token':tok.decode(original['output_ids'][j:j+1]),'decoded_prefix':before}
                violations.append(event)
            ng[tuple(hist[-2:])].add(token);hist.append(token)
        recovered=eligible&set(relaxed['prediction'])-set(original['prediction'])
        rr={'id':qid,'hop':original['stratum'],'question':original['question'],'eligible_correct_endpoints':sorted(eligible),'potentially_exposed_entities':sorted(exposed),
            'correct_endpoint_entities_with_blocked_tokens_on_relaxed_trajectory':sorted(correct_blocked),'recovered_correct_endpoints':sorted(recovered),
            'original_f1':original['metrics']['f1'],'no_ngram_f1':relaxed['metrics']['f1'],'neither_f1':g['no_rep_constraints']['metrics']['f1'],
            'n_violations':len(violations),'violations':violations}
        output.append(rr)
        if len(output)%100==0:print('TRACED',len(output),flush=True)
        for event in violations:
            if event['shared_prefix_with_original'] and event['prompt_trigram_positions'] and set(event['correct_endpoint_entities'])&recovered:
                cases.append({**rr,'event':event,'original_output':original['raw_response'],'no_ngram_output':relaxed['raw_response'],
                    'evidence':original['rendered_evidence'],'gold_answers':original['gold_answers'],'prompt':original['prompt']});break
    assert len(output)==900
    dump(a.output/'trace_per_question.json',output)
    chosen=[]
    for hop in (1,2,3):
        match=[r for r in cases if r['hop']==hop]
        if match:chosen.append(match[0])
    for r in cases:
        if len(chosen)>=3:break
        if r['id'] not in {x['id'] for x in chosen}:chosen.append(r)
    dump(a.output/'mechanism_cases.json',chosen)
    summary={'n':900,'eligible_gold_endpoint_occurrences':eligible_count,'potentially_exposed_occurrences':exposed_count,
        'questions_with_eligible_endpoints':sum(bool(r['eligible_correct_endpoints']) for r in output),
        'questions_with_potential_exposure':sum(bool(r['potentially_exposed_entities']) for r in output),
        'questions_with_counterfactual_correct_endpoint_block':sum(bool(r['correct_endpoint_entities_with_blocked_tokens_on_relaxed_trajectory']) for r in output),
        'shared_prefix_case_pool':len(cases),'selected_cases':[r['id'] for r in chosen],
        'case_rule':'first ID with shared-prefix divergence and prompt-origin forbidden correct token, one per hop if available, fill by ID',
        'subgroups_are_descriptive':True,'source_sha256':hashlib.sha256(a.records.read_bytes()).hexdigest(),
        'tokenizer_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.model_dir.glob('*token*') if f.is_file()}}
    for label,test in [('exposed',lambda r:bool(r['potentially_exposed_entities'])),('unexposed_eligible',lambda r:bool(r['eligible_correct_endpoints']) and not r['potentially_exposed_entities']),('no_correct_endpoint',lambda r:not r['eligible_correct_endpoints'])]:
        rr=[r for r in output if test(r)];summary[label]={'n':len(rr),'hop_counts':dict(__import__('collections').Counter(r['hop'] for r in rr)),
            'mean_f1_change':sum(r['no_ngram_f1']-r['original_f1'] for r in rr)/len(rr) if rr else None,
            'recovered_endpoints':sum(len(r['recovered_correct_endpoints']) for r in rr)}
    dump(a.output/'mechanism_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
