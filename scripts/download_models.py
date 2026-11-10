#!/usr/bin/env python3
"""Download selected, pinned model snapshots into a local GPU workspace."""
import argparse
from pathlib import Path
MODELS={
 'qwen':('Qwen/Qwen3-8B-AWQ','cb7d6a337aadb4d2082ed0dcef1032e4f8645194','models/Qwen3-8B-AWQ'),
 'phi':('microsoft/Phi-4-mini-instruct','cfbefacb99257ffa30c83adab238a50856ac3083','models/Phi-4-mini-instruct'),
 'e5':('intfloat/e5-small-v2','ffb93f3bd4047442299a41ebb6fa998a38507c52','revision_20260909/models/e5-small-v2')}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--model',choices=['all',*MODELS],default='all')
    ap.add_argument('--download',action='store_true',help='Actually download; otherwise show the plan')
    a=ap.parse_args()
    for name in (MODELS if a.model=='all' else [a.model]):
        repo,rev,sub=MODELS[name];dest=a.workspace.resolve()/sub
        marker=dest/'.source-revision'
        if dest.exists() and any(dest.iterdir()) and (not marker.is_file() or marker.read_text().strip()!=rev):
            raise SystemExit(f'Use an empty directory or the matching download marker: {dest}')
        print(f'{repo}@{rev} -> {dest}',flush=True)
        if a.download:
            from huggingface_hub import snapshot_download
            dest.mkdir(parents=True,exist_ok=True);marker.write_text(rev+'\n')
            snapshot_download(repo_id=repo,revision=rev,local_dir=str(dest),
                              allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py','README.md','LICENSE*'])
if __name__=='__main__':main()
