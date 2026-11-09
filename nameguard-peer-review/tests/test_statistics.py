import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paired_stats import zero_failure_upper,paired_bootstrap,total_outcome_disagreement


class StatisticsMath(unittest.TestCase):
    def test_exact_zero_failure_binomial_bound(self):
        for n in (1,10,100):
            p=zero_failure_upper(n)
            self.assertAlmostEqual((1-p)**n,.05)

    def test_score_parity_can_hide_admission_failure(self):
        result=total_outcome_disagreement([1.,1.,None])
        self.assertFalse(result['conditional_score_disagreement'])
        self.assertTrue(result['total_outcome_disagreement'])

    def test_all_rejected_not_successful_coverage(self):
        result=total_outcome_disagreement([None,None])
        self.assertTrue(result['any_rejected'])
        self.assertFalse(result['total_outcome_disagreement'])

    def test_paired_bootstrap_deterministic_and_constant_exact(self):
        self.assertEqual(paired_bootstrap([0.]*12)['upper_95'],0.)
        self.assertEqual(paired_bootstrap([1.,2.,3.]),paired_bootstrap([1.,2.,3.]))
