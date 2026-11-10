#!/usr/bin/env python3
"""Print or execute post-review controls after the primary experiments complete."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path


def link_shared_result(workspace):
    old=workspace/'revision_20260909/results/shared_pool_webqsp_all1577_B256.json'
    new=workspace/'supplement/results/shared_pool_webqsp_all1577_B256.json'
    if old.exists():
        if new.exists() and hashlib.sha256(old.read_bytes()).digest()!=hashlib.sha256(new.read_bytes()).digest():
            raise ValueError('Two different common-pool results: choose one workspace, do not merge runs.')
    else:
        if not new.is_file():raise FileNotFoundError(new)
        if json.loads(new.read_text()).get('status')!='PASS':raise ValueError('Common-pool run incomplete')
        old.parent.mkdir(parents=True,exist_ok=True);old.symlink_to(new.resolve())
    if json.loads(old.read_text()).get('status')!='PASS':raise ValueError('Common-pool run incomplete')


def jobs(workspace):
    root=workspace/'review_followup'
    return [
        ('reader',['src/run_reader_interventions.py','--project',str(workspace),'--group','all','--batch-size','8','--output',str(root/'reader_formal'),'--timeout-hours','8']),
        ('prepare-semantic',['src/prepare_semantic_controls.py','--project',str(workspace),'--output',str(root/'semantic_candidates'),'--timeout-hours','8']),
        ('semantic',['src/run_semantic_controls.py','--project',str(workspace),'--candidates',str(root/'semantic_candidates'),'--output',str(root/'semantic_formal'),'--encoder-batch-size','64','--reader-batch-size','8','--timeout-hours','8']),
        ('cost-decoder',['src/run_cost_decoder_controls.py','--project',str(workspace),'--output',str(root/'cost_decoder_formal'),'--batch-size','8','--timeout-hours','8'])]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--job',choices=['all','reader','prepare-semantic','semantic','cost-decoder'],default='all')
    ap.add_argument('--run',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();w=a.workspace.resolve()
    if a.resume and a.job=='all':ap.error('Resume one named job with unchanged inputs/configuration.')
    if a.run:link_shared_result(w)
    for name,cmd in jobs(w):
        if a.job not in ('all',name):continue
        if a.resume:cmd.append('--resume')
        print(f'[{name}] (cwd: {w})\n{shlex.join([sys.executable,*cmd])}',flush=True)
        if not a.run:continue
        if name=='prepare-semantic':
            subprocess.run([sys.executable,'scripts/verify_resources.py','--workspace',str(w),'--scope','webqsp','--include-training'],cwd=w,check=True)
        logdir=w/'logs';logdir.mkdir(exist_ok=True)
        with (logdir/f'review-{name}.log').open('a' if a.resume else 'x') as log:
            subprocess.run([sys.executable,*cmd],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not a.run:print('Plan only. Add --run to execute sequentially; data and models remain local.')


if __name__=='__main__':main()
