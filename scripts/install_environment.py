#!/usr/bin/env python3
"""Print or install pinned dependencies into the active Python environment."""
import argparse
import platform
import shlex
import subprocess
import sys

ANALYSIS = ['numpy==2.3.5', 'matplotlib==3.10.3', 'rank-bm25==0.2.2']
INFERENCE = [
    'torch==2.5.1', 'torchvision==0.20.1', 'triton==3.1.0',
    'transformers==4.51.3', 'tokenizers==0.21.1', 'accelerate==1.6.0',
    'huggingface-hub==0.30.2', 'autoawq==0.2.9', 'autoawq-kernels==0.0.9',
    'numpy==2.1.3', 'rank-bm25==0.2.2', 'safetensors==0.5.3',
    'sentencepiece==0.2.0', 'protobuf==5.29.4', 'datasets==3.5.1',
    'fsspec==2024.10.0',
]


def commands(mode):
    pip = [sys.executable, '-m', 'pip', 'install']
    if mode == 'analysis':
        return [[*pip, *ANALYSIS]]
    return [
        [*pip, 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu124'],
        [*pip, *INFERENCE],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['analysis', 'inference'], required=True)
    ap.add_argument('--install', action='store_true', help='Execute; default prints commands')
    args = ap.parse_args()
    if args.install and sys.prefix == sys.base_prefix:
        ap.error('Activate a virtual environment or invoke its Python interpreter first.')
    if args.install and args.mode == 'inference' and (
        platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64')
        or sys.version_info[:2] != (3, 12)
    ):
        ap.error('Historical inference requires Linux x86_64 and Python 3.12.')
    for command in commands(args.mode):
        print(shlex.join(command), flush=True)
        if args.install:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
