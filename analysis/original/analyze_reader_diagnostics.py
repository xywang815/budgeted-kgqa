#!/usr/bin/env python3
"""Post-hoc answer-stage diagnostics from frozen outputs; no model inference.

WebQSP uses a chain-stratified paired percentile bootstrap. The five selectors
share bootstrap question indices, but no new p values or confirmatory test family
are created. MetaQA cap flags are counts, not labels of incorrect truncation.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from analyze_revision import metrics
ROOT=Path(__file__).resolve().parent
SEED=20260914
RESAMPLES=20000
CONDITIONS=['bm25','racps','relation_density','e5_rank','e5_density']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compute():
    hashes={};caps=[]
    for budget in [128,256,512]:
        path=ROOT/f'results/qwen3_awq_test300_B{budget}.json';hashes[str(path.relative_to(ROOT))]=digest(path)
        data=json.loads(path.read_text());rows=[r for r in data['results'] if r['condition']=='ra_mmr']
        assert len(rows)==len({r['id'] for r in rows})==900
        assert all(isinstance(r['output_hit_max_new_tokens'],bool) for r in rows)
        count=sum(r['output_hit_max_new_tokens'] for r in rows)
        caps.append({'budget':budget,'condition':'ra_mmr','n':900,'output_cap':96,'reached_cap':count,'reached_cap_rate':count/900})
    path=ROOT/'supplement/results/shared_pool_webqsp_all1577_B256.json';hashes[str(path.relative_to(ROOT))]=digest(path)
    data=json.loads(path.read_text());assert data['status']=='PASS'
    records=[r for r in data['results'] if r['group']=='extension1077']
    by={(r['id'],r['condition']):r for r in records};ids=sorted({r['id'] for r in records})
    assert len(ids)==1077 and len(records)==len(by)==1077*5
    diffs=[];reader=[];endpoint=[];chains=[]
    for q in ids:
        chain='1' if by[(q,'racps')]['annotated_max_hops']==1 else '2+'
        rr=[];ee=[]
        for condition in CONDITIONS:
            r=by[(q,condition)]
            assert ('1' if r['annotated_max_hops']==1 else '2+')==chain
            rf=metrics(r['mapped_prediction_ids'],r['gold_ids'])['f1']
            ef=metrics(r['selected_endpoint_ids'],r['gold_ids'])['f1']
            assert abs(rf-r['f1'])<1e-12 and abs(ef-r['endpoint_metrics']['f1'])<1e-12
            rr.append(rf);ee.append(ef)
        reader.append(rr);endpoint.append(ee);diffs.append(np.array(rr)-np.array(ee));chains.append(chain)
    diffs=np.array(diffs);reader=np.array(reader);endpoint=np.array(endpoint);chains=np.array(chains)
    rng=np.random.default_rng(SEED);boot=np.empty((RESAMPLES,len(CONDITIONS)))
    # Preserve the observed chain counts and each question's reader/endpoint pair.
    for start in range(0,RESAMPLES,200):
        size=min(200,RESAMPLES-start);sums=np.zeros((size,len(CONDITIONS)))
        for group in ['1','2+']:
            values=diffs[chains==group];indexes=rng.integers(len(values),size=(size,len(values)))
            sums+=values[indexes].sum(axis=1)
        boot[start:start+size]=sums/len(ids)
    bounds=np.quantile(boot,[.025,.975],axis=0)
    rows=[{'condition':c,'n':len(ids),'reader_f1':float(reader[:,i].mean()),'endpoint_f1':float(endpoint[:,i].mean()),'reader_minus_endpoint':float(diffs[:,i].mean()),'ci_low':float(bounds[0,i]),'ci_high':float(bounds[1,i])} for i,c in enumerate(CONDITIONS)]
    return {'analysis':'Exploratory answer-stage diagnostics, independent of frozen selector comparison families','seed':SEED,'bootstrap_resamples':RESAMPLES,'interval':'Pointwise 95% paired percentile bootstrap; fixed official-chain stratum sizes; NumPy linear quantiles','contrast':'reader F1 minus endpoint F1 on the same selected evidence','chain_counts':{c:int(sum(chains==c)) for c in ['1','2+']},'source_sha256':hashes,'analysis_script_sha256':digest(Path(__file__).resolve()),'metaqa_output_caps':caps,'webqsp_reader_increments':rows}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    report=compute();dest=ROOT/'derived/reader_diagnostics.json'
    if args.check:
        assert json.loads(dest.read_text())==report,'Stored reader diagnostic differs from recomputation'
        print('PASS: reader increments, paired intervals, input hashes and output-cap counts exactly reproduced')
    else:
        dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        for key in ['metaqa_output_caps','webqsp_reader_increments']:
            with (ROOT/'derived'/f'{key}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(report[key][0]));w.writeheader();w.writerows(report[key])
        print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
