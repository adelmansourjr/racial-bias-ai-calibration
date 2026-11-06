"""Check the shortlist extremum claim against exhaustive finite assignments."""
from itertools import product
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_results import top_k_status


class Shortlist(unittest.TestCase):
    def test_extrema_equal_exhaustive_assignments_with_ties(self):
        scores = {'1':(-1, 0, 1), '2':(0, 1), '3':(0, 2), '4':(2, 2)}
        intervals = {key:(min(v),max(v)) for key,v in scores.items()}
        for k in range(1, len(scores)+1):
            outcomes = {key:[] for key in scores}
            for assignment in product(*scores.values()):
                ranking = sorted(zip(assignment, scores),key=lambda pair:(-pair[0],pair[1]))
                chosen = {key for _,key in ranking[:k]}
                for key in outcomes:
                    outcomes[key].append(key in chosen)
            for key, statuses in outcomes.items():
                self.assertEqual(top_k_status(intervals,key,k),(any(statuses),all(statuses)))

    def test_all_equal_with_fixed_ties_is_deterministic(self):
        intervals = {key:(0,0) for key in ('a','b','c')}
        self.assertEqual(top_k_status(intervals,'a',2),(True,True))
        self.assertEqual(top_k_status(intervals,'c',2),(False,False))
