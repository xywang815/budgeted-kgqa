#!/usr/bin/env python3
"""Link provider downloads into the fixed workspace layout; copy no datasets."""
import argparse
from pathlib import Path

def link(source, target):
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            return
        raise FileExistsError(f'Refusing to replace {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--metaqa',type=Path,help='Provider root containing kb.txt and 1-hop/2-hop/3-hop')
    ap.add_argument('--webqsp-test',type=Path)
    ap.add_argument('--webqsp-train',type=Path,help='Official train file for post-review E5 development selection')
    ap.add_argument('--neighborhoods',type=Path,help='Directory containing WebQTest-*.nxhd and WebQTrn-*.nxhd')
    a=ap.parse_args();w=a.workspace.resolve()
    if not w.is_dir():raise SystemExit('Run setup_workspace.py first.')
    if not any((a.metaqa,a.webqsp_test,a.webqsp_train,a.neighborhoods)):ap.error('Provide at least one source path.')
    if a.metaqa:
        kb=a.metaqa/'kb.txt'
        if not kb.is_file():kb=a.metaqa/'kb/kb.txt'
        link(kb,w/'data/raw/metaqa/kb/kb.txt')
        for hop in (1,2,3):
            for split in ('dev','test'):
                rel=Path(f'{hop}-hop/vanilla/qa_{split}.txt')
                link(a.metaqa/rel,w/'data/raw/metaqa'/rel)
    if a.webqsp_test:
        link(a.webqsp_test,w/'external/webqsp_official/WebQSP/data/WebQSP.test.json')
        link(a.webqsp_test,w/'data/WebQSP.test.json')
    if a.webqsp_train:
        link(a.webqsp_train,w/'external/webqsp_official/WebQSP/data/WebQSP.train.json')
    if a.neighborhoods:
        link(a.neighborhoods,w/'external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods')
    print('Linked local resources; provider files were not modified.')

if __name__=='__main__':main()
