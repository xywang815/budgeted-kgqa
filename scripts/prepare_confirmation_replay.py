#!/usr/bin/env python3
"""Verify a separately supplied review archive and prepare its exact sample.

Copies the frozen plan and 300-question manifest, never predictions, model
weights, or full datasets. Does not require an author's historical workspace.
An existing different plan or manifest is never overwritten.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

def qkey(q):
    return ' '.join(q.casefold().replace('[','').replace(']','').split())

def prepare(review,project):
    plan=review/'confirmation_plan.json'
    source=review/'独立确认/sample_manifest.json'
    manifest=json.loads(source.read_text())
    assert hashlib.sha256(plan.read_bytes()).hexdigest()==manifest['plan_sha256']
    samples=manifest['sample'];ids={r['id'] for r in samples}
    assert len(samples)==len(ids)==300
    assert all(sum(r['hop']==h for r in samples)==100 for h in (1,2,3))
    assert len({qkey(r['question']) for r in samples})==300
    assert not ids&set(manifest['excluded_ids'])
    ledger=json.loads((review/'exclusion_ledger.json').read_text())
    assert ledger['source_hashes']==manifest['exclusion_source_hashes']
    assert {r['id'] for r in ledger['rows']}==set(manifest['excluded_ids'])
    old_questions={r['normalized_question'] for r in ledger['rows']}
    assert len(old_questions)==manifest['excluded_normalized_questions']
    assert all(not qkey(r['question']) in old_questions for r in samples)
    for row in ledger['rows']:
        assert row['source_files'] and set(row['source_files'])<=set(ledger['source_hashes'])
    output=project/'review_round2/confirmation'
    for src,dst in [(plan,project/'confirmation_plan.json'),(source,output/'sample_manifest.json')]:
        if dst.exists():
            if dst.read_bytes()!=src.read_bytes():
                raise FileExistsError(f'Different existing file: {dst}')
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    return {'status':'PASS','questions':300,'exclusion_ids':len(manifest['excluded_ids']),
            'normalized_text_overlap':0,'id_overlap':0,'source_files':len(ledger['source_hashes']),
            'plan_sha256':manifest['plan_sha256'],'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'output':str(output),'ledger_scope':ledger['scope']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-dir',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.review_dir,a.project),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
