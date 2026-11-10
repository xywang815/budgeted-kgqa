"""Check score-scale invariance and differential selection against frozen code."""
import ast,importlib.util,sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kgqa_core import Edge,GraphPath,pack_in_order,rank_by_scores
from run_semantic_controls import select_indices,scaled_scores
from prepare_semantic_controls import line_costs
class Tokenizer:
    def encode(self,text,**kwargs):return text.split()

def frozen_select():
    tree=ast.parse((ROOT/'supplement/src/run_shared_pool_webqsp.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select','count')]
    def render(paths,endpoints):return '\n'.join(f'- {p.text()} => A{i+1}' for i,p in enumerate(paths)),{}
    ns={'render_labeled_evidence':render}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<frozen-selection>','exec'),ns)
    return ns['select']

class SemanticControlsTests(unittest.TestCase):
    def test_empty_pools_retain_question_without_tokenizer_call(self):
        def reject(*args,**kwargs):raise AssertionError('empty tokenization')
        self.assertEqual(line_costs(reject,[],True),[])
        for rule in ('rank','max_density_g1','minmax_density_g05'):
            self.assertEqual(select_indices([],[],[],[],rule,Tokenizer(),True),[])
    def setUp(self):
        self.tok=Tokenizer();self.paths=[GraphPath(('Topic','Answer '+str(i)),(Edge('Topic','r','Answer '+str(i)),)) for i in range(6)]
        self.scores=[.72,.95,.8,.95,.69,.9]
    def test_meta_rank_matches_frozen_packing(self):
        texts=[p.text() for p in self.paths];costs=[len(self.tok.encode('- '+s)) for s in texts]
        got=select_indices(texts,costs,[p.hops for p in self.paths],self.scores,'rank',self.tok,False,19)
        ranked=rank_by_scores(self.paths,self.scores);expected=pack_in_order(ranked,19,self.tok,standalone_costs=[costs[self.paths.index(p)] for p in ranked])
        self.assertEqual([self.paths[i] for i in got],expected)
    def test_web_original_rank_and_density_match_frozen(self):
        texts=[p.text() for p in self.paths];costs=[len(self.tok.encode('- '+s+' => A1')) for s in texts];end={p:p.endpoint for p in self.paths}
        for rule,condition in [('rank','e5_rank'),('max_density_g1','e5_density')]:
            for budget in (1,10,20,40):
                got=select_indices(texts,costs,[1]*len(texts),self.scores,rule,self.tok,True,budget)
                _,expected=frozen_select()(self.paths,self.scores,end,self.tok,budget,condition,costs,[set()]*6)
                self.assertEqual(got,expected)
    def test_minmax_retains_order_and_has_zero_baseline(self):
        scaled,gamma=scaled_scores(self.scores,'minmax_density_g05');self.assertEqual(min(scaled),0.);self.assertEqual(max(scaled),1.);self.assertEqual(gamma,.5)
        self.assertEqual(sorted(range(6),key=lambda i:self.scores[i]),sorted(range(6),key=lambda i:scaled[i]))
    def test_constant_scores_are_defined(self):
        self.assertEqual(scaled_scores([.8,.8],'minmax_density_g1'),([0.,0.],1.))
if __name__=='__main__':unittest.main()
