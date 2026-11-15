import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proposal_budget import checked_rank
from study_cases import cases


class BudgetTests(unittest.TestCase):
    def test_soundness_and_coverage_under_every_option_order(self):
        for case in cases("test"):
            for logits in itertools.permutations(range(5)):
                definitive = False
                for budget in range(1, 5):
                    report = checked_rank(case, logits, budget)
                    if report.status != "review":
                        self.assertEqual(report.status, case.expected_status)
                        self.assertEqual(report.record_ids, case.expected_ids)
                        definitive = True
                    else:
                        self.assertFalse(definitive)
                self.assertEqual((report.status, report.record_ids), (case.expected_status, case.expected_ids))

    def test_model_abstention_is_not_a_witness(self):
        case = next(cases("test"))
        low = (5, 4, 3, 2, -100)
        high = (5, 4, 3, 2, 100)
        self.assertEqual(checked_rank(case, low, 2), checked_rank(case, high, 2))

    def test_missing_proposal_does_not_create_an_admission(self):
        case = next(cases("test"))
        self.assertIsNone(checked_rank(case, None, 2))
