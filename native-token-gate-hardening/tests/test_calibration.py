import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gate
from calibration import CredentialCalibration, fit_credential_threshold
from nameguard.runtime import PublicScore


class Calibration(unittest.TestCase):
    def test_separates_a_biased_but_ordered_fitting_rubric(self):
        fitted = fit_credential_threshold([{'margin':3.,'label':False},{'margin':5.,'label':True}])
        self.assertEqual(fitted.threshold,4.)
        self.assertLess(fitted.apply(PublicScore(3.,.95)).margin,0)
        self.assertGreater(fitted.apply(PublicScore(5.,.99)).margin,0)

    def test_identical_name_scores_remain_exact(self):
        fitted = CredentialCalibration(3.)
        score = PublicScore(5.,.9)
        self.assertEqual(fitted.apply(score),fitted.apply(score))

    def test_calibration_does_not_erase_a_preexisting_name_gap(self):
        fitted = CredentialCalibration(3.)
        self.assertNotEqual(fitted.apply(PublicScore(4.,.9)),fitted.apply(PublicScore(5.,.9)))

    def test_bad_thresholds_reject(self):
        for threshold in [float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):
                CredentialCalibration(threshold)
