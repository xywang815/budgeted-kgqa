#!/usr/bin/env python3
"""Verify local benchmark bytes against original source fingerprints (no data bundled)."""
import argparse,hashlib,json
from pathlib import Path
METAQA = {'data/raw/metaqa/kb/kb.txt': '9b4d3b3066266075ad1577b4beed68e6e112e76e1a29ed2cb00a2896d2b792af', 'data/raw/metaqa/1-hop/vanilla/qa_dev.txt': '3c68fa73a14d9b9616e49f08e2d8b65e024af417708306874fe4d398634620d3', 'data/raw/metaqa/1-hop/vanilla/qa_test.txt': 'c57e9c2c4c05e5eec4503460042c09bfa3e80ea1a5ecaf361c92dd81d6c7bece', 'data/raw/metaqa/2-hop/vanilla/qa_dev.txt': 'a713ae6b4a7cdccfdda57ac99092f4114a1f8b0c4ba51b4466a15496a4ff6fbd', 'data/raw/metaqa/2-hop/vanilla/qa_test.txt': 'b1c4f1ad77f34a27e0e5a13c08b436c6f4c9a49dccd8d43639a569ee5cd906a1', 'data/raw/metaqa/3-hop/vanilla/qa_dev.txt': 'c638b99f97db93112f7a9ceec47675066cc9efc93b7ecd43e22c54ab531741d9', 'data/raw/metaqa/3-hop/vanilla/qa_test.txt': 'f4c2a801f864e80e809655a20b4b8c9904f4a2885a7f23850b2fabd8d38a6540'}
WEBQSP_SHA='856f50eb1ede43b799b10ddd0a980e72c88475b1a2259491af57e5db410ded52'
WEBQSP_TRAIN_SHA='a01f87fa7808d2e90ccdf758b8eefbb882f3f92a8f386058ad6fd153c1589ed0'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--scope',choices=['all','metaqa','webqsp'],default='all')
    ap.add_argument('--include-training',action='store_true',help='Also require official WebQSP training data and neighborhoods')
    a=ap.parse_args();w=a.workspace.resolve();checks={}
    if a.scope in ('all','metaqa'):checks.update(METAQA)
    if a.scope in ('all','webqsp'):
        checks['data/WebQSP.test.json']=WEBQSP_SHA
        checks['external/webqsp_official/WebQSP/data/WebQSP.test.json']=WEBQSP_SHA
    if a.include_training:checks['external/webqsp_official/WebQSP/data/WebQSP.train.json']=WEBQSP_TRAIN_SHA
    for name,expected in checks.items():
        p=w/name
        if not p.is_file():raise SystemExit(f'Missing source file: {p}')
        if digest(p)!=expected:raise SystemExit(f'Source fingerprint mismatch: {name}; preserve bytes and check the provider/version.')
    print(f'PASS: {len(checks)} source files match the original fingerprints.')
    if a.scope in ('all','webqsp'):
        folder=w/'external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods'
        if not folder.is_dir() or not any(folder.glob('WebQTest-*.nxhd')):
            raise SystemExit('WebQSP neighborhood directory is missing or has no WebQTest-*.nxhd files.')
        print('Neighborhood layout found; run the resource audit to check eligibility. No full-archive hash is asserted.')
    if a.include_training:
        folder=w/'external/unikqa/extracted/data/freebase_2hop/stagg.neighborhoods'
        if not any(folder.glob('WebQTrn-*.nxhd')):raise SystemExit('Training neighborhoods required for E5 development selection.')
if __name__=='__main__':main()
