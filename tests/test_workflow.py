"""Synthetic, CPU-only integration checks; no benchmark records or model downloads."""
import ast
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class WorkflowTests(unittest.TestCase):
    def test_review_plan_and_shared_result_alias(self):
        mod=module('review_plan',ROOT/'scripts/run_review_inference.py')
        plan=mod.jobs(Path('/synthetic/workspace'))
        self.assertEqual(len(plan),4)
        for name,command in plan:
            tree=ast.parse((ROOT/command[0]).read_text())
            allowed={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
                     and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
                     for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
            for token in command[1:]:
                if token.startswith('--'):self.assertIn(token,allowed,(name,token))
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);source=w/'supplement/results/shared_pool_webqsp_all1577_B256.json'
            source.parent.mkdir(parents=True);source.write_text('{"status":"RUNNING"}')
            with self.assertRaises(ValueError):mod.link_shared_result(w)
            source.write_text('{"status":"PASS"}');mod.link_shared_result(w);mod.link_shared_result(w)
            target=w/'revision_20260909/results/shared_pool_webqsp_all1577_B256.json'
            self.assertEqual(target.resolve(),source.resolve())
            target.unlink();target.write_text('{"status":"PASS","other":true}')
            with self.assertRaises(ValueError):mod.link_shared_result(w)

    def test_workspace_layout_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td)/'workspace'
            subprocess.run([sys.executable,str(ROOT/'scripts/setup_workspace.py'),'--output',str(w)],check=True,capture_output=True)
            for p in ('src/run_kgqa_inference.py','revision_20260909/src/run_shared_pool_webqsp.py',
                      'revision_20260909/vendor/kgqa_core.py','revision_20260909/README.md',
                      'scripts/make_sample_manifest.py','analyze_revision.py'):
                self.assertTrue((w/p).is_file(),p)
            self.assertEqual(list((w/'results').iterdir()),[])
            again=subprocess.run([sys.executable,str(ROOT/'scripts/setup_workspace.py'),'--output',str(w)],capture_output=True)
            self.assertNotEqual(again.returncode,0)
            self.assertFalse(list(w.rglob('*.json')))

    def test_plan_uses_real_cli_options_without_gpu_imports(self):
        plan=module('plan',ROOT/'scripts/run_experiments.py').jobs(Path('/synthetic/workspace'))
        self.assertEqual(len(plan),19)
        for _,name,command in plan:
            script=ROOT/command[0].replace('revision_20260909/','supplement/')
            tree=ast.parse(script.read_text())
            allowed={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
                     and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
                     for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
            for token in command[1:]:
                if token.startswith('--'):self.assertIn(token,allowed,(name,token))
        out=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiments.py'),'--workspace','/missing/synthetic/workspace'],capture_output=True,text=True,check=True)
        self.assertIn('Plan only',out.stdout)

    def test_webqsp_generation_cap_matches_frozen_protocol(self):
        protocol=(ROOT/'supplement/README.md').read_text()
        cap=re.search(r'max_new_tokens=(\d+)',protocol).group(1)
        jobs=module('caps',ROOT/'scripts/run_experiments.py').jobs(Path('/synthetic/workspace'))
        command=next(cmd for _,name,cmd in jobs if name=='legacy-webqsp')
        self.assertEqual(command[command.index('--max-new-tokens')+1],cap)

    def test_sample_manifest_from_synthetic_reference(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);ref=p/'reference.json';out=p/'manifest.json'
            rows=[{'id':f'synthetic-{hop}-{i}','hop':hop,'condition':'ra_mmr','line_number':i,
                   'question':'synthetic question','topic_entity':'synthetic entity'} for hop in (1,2,3) for i in range(300)]
            ref.write_text(json.dumps({'status':'PASS','configuration':{'seed':20260818},'results':rows}))
            subprocess.run([sys.executable,str(ROOT/'scripts/make_sample_manifest.py'),'--reference',str(ref),'--output',str(out)],check=True,capture_output=True)
            manifest=json.loads(out.read_text());self.assertEqual(len(manifest['questions']),900)
            self.assertFalse(manifest['gold_answers_included'])

    def test_resource_links_and_mismatch_detection(self):
        links=module('links',ROOT/'scripts/link_resources.py')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);a=p/'a';b=p/'b';a.write_text('synthetic a');b.write_text('synthetic b')
            dest=p/'workspace/link';links.link(a,dest);links.link(a,dest)
            self.assertEqual(dest.resolve(),a.resolve())
            with self.assertRaises(FileExistsError):links.link(b,dest)
            chk=subprocess.run([sys.executable,str(ROOT/'scripts/verify_resources.py'),'--workspace',str(p),'--scope','metaqa'],capture_output=True,text=True)
            self.assertNotEqual(chk.returncode,0)
            self.assertIn('Missing source file',chk.stderr)

    def test_partial_audit_inputs_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'results').mkdir()
            (p/'results/qwen3_awq_test300_B128.json').write_text('{"status":"RUNNING","results":[]}')
            out=p/'audit'
            cp=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_audit_inputs.py'),'--workspace',str(p),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(cp.returncode,0);self.assertIn('Incomplete experiment',cp.stderr)
            self.assertFalse(out.exists())

    def test_original_selector_properties(self):
        subprocess.run([sys.executable,str(ROOT/'src/test_week2_selectors.py')],check=True,capture_output=True)

    def test_missing_analysis_inputs_fail_before_computation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            cp=subprocess.run([sys.executable,str(ROOT/'scripts/run_analysis.py'),'--workspace',str(p)],capture_output=True,text=True)
            self.assertNotEqual(cp.returncode,0)
            self.assertIn('Missing local analysis inputs',cp.stderr)
            self.assertEqual(list(p.iterdir()),[])

if __name__=='__main__':unittest.main()
