import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('revision_controls', Path(__file__).resolve().parents[1] / 'analysis/revision_controls.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


class RevisionControlsTests(unittest.TestCase):
    def test_votes_count_paths_and_preserve_all_maximum_ties(self):
        self.assertEqual(controls.vote_answers(['b', 'a', 'a', 'b', 'c'], 'max_vote'), ['b', 'a'])
        self.assertEqual(controls.vote_answers([], 'max_vote'), [])

    def test_top5_has_stable_ties_and_deduplicates(self):
        self.assertEqual(controls.vote_answers(['b', 'c', 'd', 'e', 'f', 'g', 'a', 'a'], 'vote_top5'), ['a', 'b', 'c', 'd', 'e'])

    def test_path_endpoint_extraction_respects_direction(self):
        row = {'id': 'synthetic', 'selected_paths': 2, 'selected_evidence': ['Topic <--[acted in]-- Film --[written by]--> Writer', 'Topic --[starred actors]--> Actor'], 'selected_endpoints': ['Writer', 'Actor']}
        self.assertEqual(controls.path_endpoints(row), ['writer', 'actor'])

    def test_filtering_can_remove_a_correct_nonendpoint_answer(self):
        self.assertEqual(controls.metrics(['outside'], ['outside'])['f1'], 1.)
        self.assertEqual(controls.metrics([], ['outside'])['f1'], 0.)


if __name__ == '__main__':
    unittest.main()
