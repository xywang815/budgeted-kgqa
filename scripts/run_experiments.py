#!/usr/bin/env python3
"""Print or execute the paper's ordered experiment commands; no runs by default."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

STAGES=('preprocess','tuning','metaqa','components','phi','webqsp','shared-smoke','shared-webqsp')

def jobs(workspace):
    result=[]
    def add(stage,name,script,*args):result.append((stage,name,[script,*map(str,args)]))
    data=['--data-dir','data/processed/metaqa','--model-dir','models/Qwen3-8B-AWQ']
    settings=['--split','test','--sample-per-hop','300','--max-paths','100000',
              '--ba-mmr-lambda','.9','--ba-cost-exponent','.5',
              '--ra-mmr-lambda','.9','--ra-cost-exponent','1',
              '--max-new-tokens','96','--checkpoint-every','5']
    reference='results/qwen3_awq_test300_B256.json'
    manifest='results/week2_sample_ids_metaqa_test_seed20260818.json'
    official='external/webqsp_official/WebQSP/data/WebQSP.test.json'
    graph='external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods'
    audit='results/week2_webqsp_resource_audit.json'
    add('preprocess','verify-metaqa','scripts/verify_resources.py','--workspace',workspace,'--scope','metaqa')
    add('preprocess','prepare-metaqa','src/prepare_metaqa.py','--data-dir','data/raw/metaqa',
        '--output-dir','data/processed/metaqa','--manifest','results/preprocess_manifest.json')
    add('tuning','development-grid','src/tune_mmr.py',*data,'--split','dev','--sample-per-hop','50',
        '--seed','42','--budgets','128','256','512','--mmr-lambdas','.3','.5','.7','.9',
        '--cost-exponents','0','.5','1','--max-paths','100000','--workers','8','--output','results/tune_mmr_dev50.json')
    for budget,seed in [(128,20260818),(256,20260818),(512,20260818),(256,20260819),(256,20260820)]:
        tail='' if seed==20260818 else f'_seed{seed}'
        add('metaqa',f'qwen-b{budget}-s{seed}','src/run_kgqa_inference.py',*data,*settings,
            '--seed',seed,'--budget',budget,'--output',f'results/qwen3_awq_test300_B{budget}{tail}.json')
    add('components','sample-manifest','scripts/make_sample_manifest.py','--reference',reference,'--output',manifest)
    component=[*data,*settings,'--seed','20260818','--budget','256','--reference-results',reference,
               '--output','results/clean_component_ablation_qwen_B256.json']
    add('components','component-preflight','src/run_component_ablation.py',*component,'--preflight-only')
    add('components','clean-components','src/run_component_ablation.py',*component)
    add('components','historical-component-diagnostics','src/run_week2_components.py',*data,
        '--reference-results',reference,'--sample-manifest',manifest,'--split','test','--sample-per-hop','300',
        '--seed','20260818','--budget','256','--max-paths','100000','--standard-mmr-lambda','.9',
        '--max-new-tokens','96','--checkpoint-every','10','--output','results/week2_components_qwen_test300_B256.json')
    add('phi','phi-reselection','src/run_kgqa_inference_multimodel.py',
        '--data-dir','data/processed/metaqa','--model-dir','models/Phi-4-mini-instruct',
        '--model-id','microsoft/Phi-4-mini-instruct','--model-revision','cfbefacb99257ffa30c83adab238a50856ac3083',
        '--model-dtype','bfloat16','--prompt-family','generic',*settings,'--seed','20260818','--budget','256',
        '--output','results/phi4mini_test300_B256_seed20260818.json')
    add('phi','phi-fixed-evidence','src/run_shared_evidence_reader.py','--reference-results',reference,
        '--sample-manifest',manifest,'--selector-tokenizer-dir','models/Qwen3-8B-AWQ',
        '--reader-model-dir','models/Phi-4-mini-instruct','--sample-per-hop','300','--seed','20260818',
        '--max-new-tokens','96','--checkpoint-every','5','--output','results/week2_shared_evidence_phi_qwenpaths_test300_B256.json')
    add('webqsp','verify-webqsp','scripts/verify_resources.py','--workspace',workspace,'--scope','webqsp')
    add('webqsp','resource-audit','src/audit_webqsp_resource.py','--official-test',official,'--subgraph-dir',graph,
        '--output-json',audit,'--output-csv','results/week2_webqsp_resource_audit.csv')
    add('webqsp','legacy-webqsp','src/run_webqsp_replication.py','--official-test',official,'--audit',audit,
        '--subgraph-dir',graph,'--model-dir','models/Qwen3-8B-AWQ','--sample-size','500','--seed','20260818',
        '--budget','256','--max-paths','100000','--rerank-cap','20000','--max-new-tokens','32',
        '--checkpoint-every','5','--output','results/week2_webqsp_qwen_test500_B256.json')
    shared=['--project',str(workspace),'--protocol','revision_20260909/README.md','--batch-size','64']
    add('shared-smoke','synthetic-smoke','revision_20260909/src/run_shared_pool_webqsp.py',*shared,
        '--smoke','--output','supplement/results/synthetic_smoke.json')
    add('shared-webqsp','formal-shared-webqsp','revision_20260909/src/run_shared_pool_webqsp.py',*shared,
        '--output','supplement/results/shared_pool_webqsp_all1577_B256.json')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--stage',choices=['all',*STAGES],default='all')
    ap.add_argument('--job',help='Restrict to one printed job name; useful after an interruption')
    ap.add_argument('--run',action='store_true',help='Execute; the default prints commands only')
    a=ap.parse_args();w=a.workspace.resolve()
    chosen=[j for j in jobs(w) if (a.stage=='all' or j[0]==a.stage) and (not a.job or j[1]==a.job)]
    if not chosen:ap.error('No matching job.')
    for stage,name,cmd in chosen:
        print(f'[{stage}/{name}] (cwd: {w})\n{shlex.join([sys.executable,*cmd])}',flush=True)
        if not a.run:continue
        if not (w/cmd[0]).is_file():raise SystemExit('Missing workspace code; run setup_workspace.py first.')
        for flag in ('--output','--output-json','--output-csv','--manifest'):
            if flag in cmd and (w/cmd[cmd.index(flag)+1]).exists():
                raise SystemExit(f'{name}: output already exists. Do not overwrite completed experiments.')
        (w/'logs').mkdir(exist_ok=True)
        with (w/'logs'/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,*cmd],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not a.run:print('Plan only. Add --run on the prepared GPU server to execute.')
if __name__=='__main__':main()
