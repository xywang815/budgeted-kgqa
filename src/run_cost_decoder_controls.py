#!/usr/bin/env python3
"""Matched 2x2 cost-by-decoding control, prompted by the initial reader intervention.

Reuses the unchanged fixed-path engine; generates all four cells anew. No condition
is chosen using this control's outputs, and no historical answers are reused.
"""
import argparse,json,sys
from pathlib import Path
import run_reader_interventions as engine

CONDITIONS=('original','no_cost_original','no_rep_constraints','no_cost_no_constraints')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--limit-per-stratum',type=int,default=0)
    ap.add_argument('--timeout-hours',type=float,default=8.);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=a.output.resolve()
    receipt={'driver_sha256':engine.sha(__file__),'engine_sha256':engine.sha(engine.__file__),
        'conditions':CONDITIONS,'expected_full_rows':3600,'original_paths_and_B256_preserved':True,
        'scope':'Exploratory second-wave control, specified after observing the initial trigram-removal effect; frozen before its own outcomes.'}
    path=out.parent/(out.name+'_driver.json');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(receipt))
    else:path.write_text(json.dumps(receipt,indent=2)+'\n')
    engine.META_CONDITIONS=CONDITIONS
    sys.argv=[sys.argv[0],'--project',str(a.project),'--output',str(out),'--group','meta256',
        '--batch-size',str(a.batch_size),'--limit-per-stratum',str(a.limit_per_stratum),'--timeout-hours',str(a.timeout_hours)]
    if a.resume:sys.argv.append('--resume')
    engine.main()

if __name__=='__main__':main()
