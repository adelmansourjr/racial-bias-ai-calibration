"""Research-claim negative controls independent of the model implementation."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import scientific_claims as s


class ClaimBoundaries(unittest.TestCase):
    def test_equal_behavior_can_retain_name(self):
        r = s.behavioral_equality_is_not_erasure()
        self.assertTrue(r['equal_scores'] and r['names_still_recoverable'])

    def test_linear_criterion_is_not_unrestricted_erasure(self):
        r = s.zero_covariance_is_not_independence()
        self.assertEqual(r, {'covariance':'0', 'threshold_accuracy':'3/4', 'nonlinear_accuracy':'1'})

    def test_finite_success_can_miss_an_unseen_name(self):
        r = s.finite_test_is_not_universal()
        self.assertTrue(r['observed_equal'] and r['unseen_violation'])

    def test_agreement_can_share_an_omission(self):
        r = s.agreement_is_not_truth()
        self.assertTrue(r['two_reviews_agree'])
        self.assertFalse(r['identity_coverage_complete'])

    def test_external_and_internal_canonicalization_match(self):
        self.assertTrue(s.canonical_baseline_equivalence()['same_outputs'])
