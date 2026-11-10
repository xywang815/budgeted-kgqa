"""CPU checks for path-preserving interventions and strict answer-code scoring."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('interventions',Path(__file__).resolve().parents[1]/'src/run_reader_interventions.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReaderInterventionsTests(unittest.TestCase):
    def test_eos_padding_is_not_counted_as_generated_answers(self):
        self.assertEqual(m.trim_generated_ids([7,9,9,9],[9,10]),[7,9])
        self.assertEqual(m.trim_generated_ids([7,8],9),[7,8])
    def test_sort_preserves_members_and_duplicates(self):
        paths=['z','a','z'];self.assertEqual(m.ordered_paths(paths,'canonical'),['a','z','z']);self.assertEqual(paths,['z','a','z'])
    def test_terminal_entity_handles_inverse_relation(self):
        self.assertEqual(m.terminal('Topic <--[acted in]-- Film --[written by]--> Writer'),'Writer')
    def test_invalid_code_is_retained_as_false_positive(self):
        self.assertEqual(m.strict_codes('A1|A999|a1',{'a1':'answer'}),['answer','__invalid_code__a999'])
    def test_generation_controls_change_separately(self):
        base=m.generation_config('original','MetaQA')
        for condition,key,value in [('cap192','max_new_tokens',192),('no_rep_penalty','repetition_penalty',1.0),('no_ngram_block','no_repeat_ngram_size',0)]:
            expected=dict(base);expected[key]=value;self.assertEqual(m.generation_config(condition,'MetaQA'),expected)
        self.assertEqual(m.generation_config('cap64','WebQSP')['max_new_tokens'],64)
        expected=dict(base,repetition_penalty=1.0,no_repeat_ngram_size=0)
        for condition in ('no_rep_constraints','no_cost_no_constraints'):
            self.assertEqual(m.generation_config(condition,'MetaQA'),expected)

if __name__=='__main__':unittest.main()
