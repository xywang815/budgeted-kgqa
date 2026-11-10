#!/usr/bin/env python3
"""Verify the active CPU-analysis or historical Linux/CUDA inference environment."""
import argparse,importlib,importlib.metadata as md,platform,sys
CPU={'numpy':'2.3.5','matplotlib':'3.10.3','rank-bm25':'0.2.2'}
GPU={'torch':'2.5.1','torchvision':'0.20.1','triton':'3.1.0','transformers':'4.51.3',
     'tokenizers':'0.21.1','accelerate':'1.6.0','huggingface-hub':'0.30.2',
     'autoawq':'0.2.9','autoawq-kernels':'0.0.9','numpy':'2.1.3','rank-bm25':'0.2.2'}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['analysis','inference'],required=True)
    a=ap.parse_args();expected=CPU if a.mode=='analysis' else GPU
    if a.mode=='inference' and (platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64') or sys.version_info[:2]!=(3,12)):
        raise SystemExit('Historical inference requires Linux x86_64 and Python 3.12.')
    for package,wanted in expected.items():
        actual=md.version(package)
        if actual.split('+')[0]!=wanted:raise SystemExit(f'{package}: expected {wanted}, got {actual}')
    modules=['numpy','rank_bm25','matplotlib'] if a.mode=='analysis' else ['torch','transformers','awq','awq_ext']
    for module in modules:importlib.import_module(module)
    if a.mode=='inference':
        import torch
        if not torch.cuda.is_available() or torch.version.cuda!='12.4':raise SystemExit('CUDA 12.4 runtime/GPU is unavailable.')
        if not torch.cuda.is_bf16_supported():raise SystemExit('Phi requires bfloat16 support.')
        print(torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory)
    print(f'PASS: {a.mode} versions and imports.')
if __name__=='__main__':main()
