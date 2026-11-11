#!/usr/bin/env python3
"""Reparse saved predictions and partition answer occurrences by endpoint membership.

Counts are question-answer pairs; P/R/F1 are question-macro means. Endpoint
membership is a location diagnostic, not an attribution of model knowledge.
"""
import argparse,csv,gzip,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from analyze_reader_interventions import parse_names,normalize,csvout,dump,cluster_interval
from revision_controls import metrics,interval

def records(f):
    with gzip.open(f,'rt') if f.suffix=='.gz' else f.open() as h:
        for line in h:yield json.loads(line)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);per=[];source_hashes={}
    for matrix,rel in [('reader','reader_formal/records.jsonl.gz'),('semantic','semantic_formal/reader_records.jsonl.gz')]:
        f=a.source/rel
        if not f.exists(): f=f.with_suffix('')
        source_hashes[str(f.relative_to(a.source))]=hashlib.sha256(f.read_bytes()).hexdigest()
        for r in records(f):
            if r['dataset']!='MetaQA' or r.get('selection_budget',256)!=256:continue
            if matrix=='reader' and r['condition'] not in ('original','no_ngram_block','no_rep_constraints','labeled_names_no_constraints','labeled_codes_no_constraints'):continue
            pred=r['prediction'];gold=set(map(normalize,r['gold_answers']));end=set(r.get('selected_endpoints',r.get('endpoints',[])))
            if not r['condition'].startswith('labeled_codes'):assert pred==parse_names(r['raw_response'])
            else:
                import re
                assert pred==list(dict.fromkeys(r['answer_code_map'].get(c.lower(),'__invalid_code__'+c.lower()) for c in re.findall(r'(?i)\bA\d+\b',r['raw_response'])))
            pred=set(pred);score=metrics(pred,gold);assert abs(score['f1']-r['metrics']['f1'])<1e-12
            filt=metrics(pred&end,gold);ep=metrics(end,gold)
            counts={'correct_in':len(pred&gold&end),'correct_out':len((pred&gold)-end),'wrong_in':len((pred-gold)&end),'wrong_out':len(pred-gold-end),
                'gold_in':len(gold&end),'gold_out':len(gold-end),'predicted':len(pred)}
            assert sum(counts[x] for x in ['correct_in','correct_out','wrong_in','wrong_out'])==len(pred)
            per.append({'matrix':matrix,'selector':r['selector'],'condition':r['condition'],'id':r['id'],'hop':r['stratum'],**score,'filtered_f1':filt['f1'],'endpoint_f1':ep['f1'],**counts,
                'has_correct_out':int(counts['correct_out']>0),'has_wrong_out':int(counts['wrong_out']>0),'has_out':int(bool(pred-end)),
                'cap_hit':int(r['hit_generation_cap']),'evidence_tokens':r['evidence_tokens']})
    grouped=defaultdict(list)
    for r in per:grouped[r['matrix'],r['selector'],r['condition']].append(r)
    means=[];contrasts=[]
    countkeys=['correct_in','correct_out','wrong_in','wrong_out','gold_in','gold_out','predicted','has_correct_out','has_wrong_out','has_out','cap_hit']
    for (matrix,sel,cond),rr in sorted(grouped.items()):
        assert len(rr)==900 and len({r['id'] for r in rr})==900
        row={'matrix':matrix,'selector':sel,'condition':cond,'n':len(rr)}
        row.update({k:float(np.mean([r[k] for r in rr])) for k in ['precision','recall','f1','filtered_f1','endpoint_f1','evidence_tokens']})
        row.update({k:sum(r[k] for r in rr) for k in countkeys});means.append(row)
        contrasts.append({'matrix':matrix,'selector':sel,'condition':cond,'contrast':'filtered_minus_raw',**interval([r['filtered_f1']-r['f1'] for r in rr],[r['hop'] for r in rr])})
    csvout(a.output/'answer_provenance_per_question.csv',per);csvout(a.output/'answer_provenance_summary.csv',means);dump(a.output/'answer_filter_effects.json',contrasts)
    dump(a.output/'manifest.json',{'status':'PASS','source_sha256':source_hashes,'questions_per_cell':900,'unit':'question-answer occurrence, not unique entity across questions','macro_metrics':True,'output_cap':96,'endpoint_membership_not_causal_attribution':True})

if __name__=='__main__':main()
