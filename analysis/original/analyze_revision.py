#!/usr/bin/env python3
"""Derive revision tables from preserved per-question outputs, without model calls."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,re,unicodedata
import numpy as np
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';OUT=ROOT/'derived'

def read(n):return json.loads((RESULTS/n).read_text())
def normalize(s):
 s=unicodedata.normalize('NFKC',str(s));s=re.sub(r'^\s*(?:answer|answers)\s*:\s*','',s,flags=re.I)
 s=re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',s)
 return ' '.join(s.strip().strip('`\"\'[](){}').rstrip('.。').casefold().split())
def metrics(pred,gold):
 pred=list(dict.fromkeys(pred));p,g=set(pred),set(gold);hit=len(p&g)
 return {'precision':hit/len(p) if p else 0.,'recall':hit/len(g) if g else 0.,
  'f1':2*hit/(len(p)+len(g)) if p or g else 0.,'hits_at_1':float(bool(pred) and pred[0] in g),'exact_set_match':float(p==g)}
def mean(rs,k):return float(np.mean([r[k] for r in rs]))
def compare(diffs,groups=None,boot=10000,perm=0):
 d=np.asarray(diffs,dtype=float);n=len(d);rng=np.random.default_rng(20260909)
 if not n:return {'n':0}
 groups=np.zeros(n) if groups is None else np.asarray(groups);samples=[]
 for start in range(0,boot,200):
  size=min(200,boot-start);s=np.zeros(size)
  for g in np.unique(groups):
   x=d[groups==g];s+=x[rng.integers(len(x),size=(size,len(x)))].sum(axis=1)
  samples.extend(s/n)
 ci=np.quantile(samples,[.025,.975]);result={'n':n,'difference':float(d.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
 if perm:
  exceed=0;threshold=abs(d.mean())-1e-12
  for start in range(0,perm,200):
   size=min(200,perm-start);sign=rng.integers(0,2,size=(size,n))*2-1
   exceed+=int(np.count_nonzero(abs((sign*d).mean(axis=1))>=threshold))
  result['p']=(exceed+1)/(perm+1)
 return result

def holm(rows,pkey='p',outkey='holm_p'):
 ordered=sorted(range(len(rows)),key=lambda i:rows[i][pkey]);running=0.
 for rank,i in enumerate(ordered):
  running=max(running,min(1.,(len(rows)-rank)*rows[i][pkey]));rows[i][outkey]=running

def save_csv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def summary_rows(records,groupfields,endpointfield=None):
 groups=defaultdict(list)
 for r in records:groups[tuple(r[k] for k in groupfields)].append(r)
 out=[]
 for keys,rs in groups.items():
  d=dict(zip(groupfields,keys));d.update(n=len(rs),f1=mean(rs,'f1'))
  for k in ['evidence_answer_recall','evidence_tokens','total_input_tokens','retrieval_seconds','selection_seconds','generation_seconds','method_retrieval_seconds','output_tokens','output_hit_max_new_tokens']:
   if k in rs[0]:d[k]=mean(rs,k)
  if endpointfield:d['endpoint_f1']=float(np.mean([r[endpointfield]['f1'] for r in rs]))
  out.append(d)
 return out

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify-original-examples',action='store_true',help='Require the original archived prediction, not a fresh model output')
 args=parser.parse_args()
 from verify_rebuilt_statistics import require_if_rebuilt
 require_if_rebuilt()
 OUT.mkdir(exist_ok=True);report={'analysis_seed':20260909,'provenance':{},'primary':[],'endpoint_reference':[],'heterogeneity':[],'raw_vs_bm25':[],'simplified_metaqa':[]}
 metaqa={}
 for seed in [20260818,20260819,20260820]:
  suffix='' if seed==20260818 else f'_seed{seed}'
  d=read('qwen3_awq_test300_B256'+suffix+'.json');metaqa[seed]=d
  by={(r['hop'],r['id'],r['condition']):r for r in d['results']}
  for hop in [1,2,3]:
   pairs=[(r,by[(r['hop'],r['id'],'bm25')]) for r in d['results'] if r['condition']=='ra_mmr' and r['hop']==hop]
   # Within-hop fixed descriptive bins, not a causal test or multiplicity family.
   for feature,binfun in [('candidate_paths',lambda x:'1-100' if x<=100 else ('101-1000' if x<=1000 else '>1000')),
     ('gold_count',lambda x:'1' if x==1 else ('2-5' if x<=5 else '>5')),
     ('baseline_evidence_recall',lambda x:'0' if x==0 else ('1' if x==1 else '(0,1)'))]:
    groups=defaultdict(list)
    for r,b in pairs:
     v=len(r['gold_answers']) if feature=='gold_count' else (b['evidence_answer_recall'] if feature=='baseline_evidence_recall' else r[feature])
     groups[binfun(v)].append(r['f1']-b['f1'])
    for label,values in groups.items():
     report['heterogeneity'].append({'seed':seed,'hop':hop,'feature':feature,'bin':label,**compare(values,boot=10000)})
 primary_pairs=[]
 for budget in [128,256,512]:
  fn=f'qwen3_awq_test300_B{budget}.json';d=read(fn);report['provenance'][fn]=hashlib.sha256((RESULTS/fn).read_bytes()).hexdigest()
  for r in d['results']:r['budget']=budget
  report['primary']+=summary_rows(d['results'],['budget','condition'])
  stats=read(f'stats_qwen3_awq_test300_B{budget}.json')
  for r in stats['comparisons']:
   if r['scope']=='all-hops' and r['metric']=='raw_f1':primary_pairs.append({'budget':budget,**r,'p':r['paired_permutation_p']})
  for c in ['bm25','ra_mmr']:
   for hop in [0,1,2,3]:
    rows=[r for r in d['results'] if r['condition']==c and (not hop or r['hop']==hop)]
    ep=[metrics([normalize(x) for x in r['selected_endpoints']],[normalize(x) for x in r['gold_answers']])['f1'] for r in rows]
    report['endpoint_reference'].append({'budget':budget,'condition':c,'hop':hop or 'all','reader_f1':mean(rows,'f1'),
      'endpoint_f1':float(np.mean(ep)),**compare([x-r['f1'] for x,r in zip(ep,rows)],[r['hop'] for r in rows])})
 holm(primary_pairs);report['primary_pairs']=primary_pairs
 clean=read('clean_component_ablation_qwen_B256.json')['results'];primary=metaqa[20260818]['results']
 report['simplified_metaqa']=summary_rows([r for r in primary if r['condition'] in ['bm25','ra_mmr']]+clean,['condition'])
 week2=read('week2_components_qwen_test300_B256.json')['results']
 report['simplified_metaqa']+=summary_rows([r for r in week2 if r['condition']=='relation_only'],['condition'])
 bm={(r['hop'],r['id']):r for r in primary if r['condition']=='bm25'}
 for c in ['racps_raw_selector_labels','racps_raw_end_to_end']:
  rs=[r for r in clean if r['condition']==c]
  report['raw_vs_bm25'].append({'condition':c,**compare([r['f1']-bm[(r['hop'],r['id'])]['f1'] for r in rs],[r['hop'] for r in rs],perm=50000)})
 holm(report['raw_vs_bm25'])
 official=json.loads((ROOT/'data/WebQSP.test.json').read_text());questions={q['QuestionId']:q for q in official['Questions']}
 web=read('week2_webqsp_qwen_test500_B256.json');corrected=[]
 for r in web['results']:
  if r['condition']=='direct':
   aliases={}
   for p in questions[r['id']]['Parses']:
    if not p.get('TopicEntityMid') or not p.get('InferentialChain') or not p.get('Answers'):continue
    for a in p['Answers']:
     ident=str(a.get('AnswerArgument') or '')
     if ident.startswith('/'):ident=ident[1:].replace('/','.')
     if ident:
      aliases[normalize(ident)]=ident
      if a.get('EntityName'):aliases[normalize(a['EntityName'])]=ident
   pred=[aliases.get(normalize(p),'__unmatched_name__'+normalize(p)) for p in r['parsed_predictions']]
  else:
   pred=[r['answer_code_map'].get(c.lower(),'__invalid_code__'+c.lower()) for c in re.findall(r'(?i)\bA\d+\b',r['raw_response'])]
  nr={**r,'legacy_metrics':{k:r[k] for k in metrics([],[])},'legacy_mapped_prediction_ids':r['mapped_prediction_ids'],
    'mapped_prediction_ids':list(dict.fromkeys(pred)),**metrics(pred,r['gold_ids'])}
  nr['endpoint_metrics']=metrics(r['selected_endpoint_ids'],r['gold_ids']);nr['chain']='1' if r['annotated_max_hops']==1 else '2+'
  corrected.append(nr)
 correction={'status':'PASS','source_sha256':hashlib.sha256((RESULTS/'week2_webqsp_qwen_test500_B256.json').read_bytes()).hexdigest(),
   'scoring':'Unmatched predictions preserved. Direct uses official answer-alias matching as scoring only, not an independent general entity linker. Direct is excluded from the comparative coded-evidence table.',
   'results':corrected}
 (OUT/'webqsp_corrected.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2)+'\n')
 report['webqsp_legacy']=summary_rows(corrected,['condition'],endpointfield='endpoint_metrics')
 report['webqsp_legacy_strata']=summary_rows(corrected,['chain','condition'],endpointfield='endpoint_metrics')
 changed=[r for r in corrected if abs(r['f1']-r['legacy_metrics']['f1'])>1e-12];report['scoring_changed']={c:sum(r['condition']==c for r in changed) for c in ['direct','bfs','bm25','ra_mmr','oracle']}
 report['webqsp_cap_strata']=[];by={(r['id'],r['condition']):r for r in corrected}
 for label,pred in [('at_most_20k',lambda r:r['candidate_paths']<=20000),('above_20k',lambda r:r['candidate_paths']>20000),('truncated_100k',lambda r:r['candidate_truncated'])]:
  rs=[r for r in corrected if r['condition']=='ra_mmr' and pred(r)]
  report['webqsp_cap_strata'].append({'stratum':label,**compare([r['f1']-by[(r['id'],'bm25')]['f1'] for r in rs])})
 supplemental=ROOT/'supplement/results/shared_pool_webqsp_all1577_B256.json'
 if supplemental.exists():
  s=json.loads(supplemental.read_text());assert s['status']=='PASS','Do not analyze partial formal scores'
  expected={(q,c) for q in s['configuration']['ids'] for c in s['configuration']['conditions']}
  actual=[(r['id'],r['condition']) for r in s['results']];assert len(actual)==len(set(actual))==7885 and set(actual)==expected
  assert all(r['evidence_tokens']<=256 for r in s['results'])
  for r in s['results']:r['chain']='1' if r['annotated_max_hops']==1 else '2+'
  report['supplement']=summary_rows(s['results'],['group','condition'],endpointfield='endpoint_metrics')
  report['supplement_strata']=summary_rows(s['results'],['group','chain','condition'],endpointfield='endpoint_metrics')
  report['supplement_all']=summary_rows(s['results'],['condition'],endpointfield='endpoint_metrics')
  report['supplement_primary_pairs']=[];by={(r['id'],r['condition']):r for r in s['results']}
  rs=[r for r in s['results'] if r['condition']=='racps' and r['group']=='extension1077']
  for c in ['bm25','relation_density','e5_rank']:
   report['supplement_primary_pairs'].append({'target':'racps','baseline':c,**compare([r['f1']-by[(r['id'],c)]['f1'] for r in rs], [r['chain'] for r in rs],boot=20000,perm=50000)})
  holm(report['supplement_primary_pairs'])
  report['supplement_checks']={'n_questions':len(s['questions']),'n_records':len(actual),'budget_violations':0,'sha256':hashlib.sha256(supplemental.read_bytes()).hexdigest(),
    'extension_n':len(rs),'created_utc':s['created_utc'],'completed_utc':s['completed_utc']}
  report['pool_audit']=[]
  for g in ['legacy500','extension1077']:
   qs=[q for q in s['questions'] if q['group']==g]
   for feature in ['candidate_paths','common_pool_paths','candidate_recall','common_pool_recall','bfs20k_recall','bm25_20k_recall','relation20k_recall','shared_preparation_seconds','e5_encoding_seconds','e5_truncated_texts']:
    report['pool_audit'].append({'group':g,'feature':feature,'mean':mean(qs,feature),'n':len(qs)})
 # Regression fixtures: adding a wrong prediction must reduce precision and F1.
 assert metrics(['gold','wrong'],['gold'])['f1']==2/3 and metrics(['wrong'],['gold'])['f1']==0
 if args.verify_original_examples:
  assert next(r for r in corrected if r['id']=='WebQTest-23' and r['condition']=='direct')['f1']==2/3
 report['scoring_tests']='PASS'
 for name,rows in report.items():
  if isinstance(rows,list) and rows and isinstance(rows[0],dict):save_csv(name,rows)
 (OUT/'revision_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('PASS',list(report),'raw contrasts',report['raw_vs_bm25'])
if __name__=='__main__':main()
