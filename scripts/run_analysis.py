#!/usr/bin/env python3
"""Run the full paper analysis from completed local experiments, without model calls."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--stage',choices=['all','statistics','derived','audit','review-controls','figures'],default='all')
    ap.add_argument('--verify-original-examples',action='store_true')
    a=ap.parse_args();w=a.workspace.resolve();strict=['--verify-original-examples'] if a.verify_original_examples else []
    if a.stage=='all':
        names=[f'results/qwen3_awq_test300_B{b}.json' for b in (128,256,512)]
        names += [f'results/qwen3_awq_test300_B256_seed{s}.json' for s in (20260819,20260820)]
        names += ['results/'+n+'.json' for n in (
            'clean_component_ablation_qwen_B256','week2_components_qwen_test300_B256',
            'phi4mini_test300_B256_seed20260818','week2_shared_evidence_phi_qwenpaths_test300_B256',
            'week2_webqsp_qwen_test500_B256')]
        names += ['supplement/results/shared_pool_webqsp_all1577_B256.json','data/WebQSP.test.json']
        missing=[name for name in names if not (w/name).is_file()]
        if missing:raise SystemExit('Missing local analysis inputs:\n'+'\n'.join(missing))
        for name in names[:-1]:
            if json.loads((w/name).read_text()).get('status')!='PASS':
                raise SystemExit(f'Experiment has not completed: {name}')
    def run(script,*args,cwd=ROOT):
        print(f'Running {script}',flush=True)
        subprocess.run([sys.executable,str(script),*map(str,args)],cwd=cwd,check=True)
    if a.stage in ('all','statistics'):
        run(ROOT/'scripts/build_statistics.py','--workspace',w)
    if a.stage in ('all','derived'):
        for script in ('analyze_revision.py','analyze_reader_diagnostics.py','analyze_exploratory_clusters.py'):
            run(w/script,*(strict if script=='analyze_revision.py' else []),cwd=w)
    if a.stage in ('all','audit'):
        if (w/'audit/inputs').exists() or (w/'audit/results').exists():
            raise SystemExit('Audit inputs/outputs already exist; preserve them or use a new analysis workspace.')
        run(ROOT/'scripts/prepare_audit_inputs.py','--workspace',w,'--output',w/'audit/inputs')
        run(ROOT/'analysis/audit/recompute.py','--inputs',w/'audit/inputs','--output',w/'audit/results',*strict)
    if a.stage in ('all','review-controls'):
        args=[]
        for budget in (128,256,512):
            args += ['--metaqa',w/f'results/qwen3_awq_test300_B{budget}.json']
        run(ROOT/'analysis/revision_controls.py',*args,
            '--components',w/'results/clean_component_ablation_qwen_B256.json',
            '--webqsp',w/'supplement/results/shared_pool_webqsp_all1577_B256.json',
            '--webqsp-official',w/'data/WebQSP.test.json','--output',w/'review_controls')
        names=sorted((w/'results').glob('*.json'))
        # Aggregate experiment outputs only; statistical and manifest JSON are excluded.
        inputs=[]
        for path in names + [w/'supplement/results/shared_pool_webqsp_all1577_B256.json']:
            obj=json.loads(path.read_text())
            if isinstance(obj,dict) and obj.get('results') and isinstance(obj['results'],list):
                inputs += ['--input',f'{path.stem}={path}']
        run(ROOT/'analysis/export_full_metrics.py',*inputs,'--output',w/'review_controls/full_experiment_metrics.csv')
    if a.stage in ('all','figures'):
        run(w/'build_visuals.py',cwd=w)
        run(ROOT/'analysis/audit/figures/gen_fig_component_stages.py','--results',w/'audit/results/component_stages.json',
            '--output-dir',w/'manuscript/figures')
        if (w/'review_controls/output_interface.json').is_file():
            run(ROOT/'analysis/gen_revision_controls.py','--analysis-dir',w/'review_controls',
                '--output',w/'manuscript/figures')
    print('Completed requested analysis stages. No model calls.')
if __name__=='__main__':main()
