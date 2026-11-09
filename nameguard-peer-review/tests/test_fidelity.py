import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fidelity_audit import audit,validate_cohort
from fixtures import contexts,GPU_NAMES


class FidelityMath(unittest.TestCase):
    def test_nonfinite_or_boolean_scores_reject(self):
        for values,constant in (((float('nan'),),0.),((1.,),float('inf')),((True,),0.)):
            with self.assertRaises(ValueError):audit(values,constant)

    def test_variance_decomposition(self):
        for constant in (-3.,0.,4.):
            r=audit((-2.,1.,3.),constant)
            self.assertAlmostEqual(r['measured_mse'],r['minimum_mse']+r['constant_offset_squared'])

    def test_midpoint_minimax_and_majority_decisions(self):
        values=(-2.,1.,3.)
        r=audit(values,.5)
        self.assertEqual(max(abs(v-.5) for v in values),r['minimum_uniform_error'])
        self.assertEqual(r['minimum_binary_decision_changes'],1)
        self.assertEqual(r['measured_binary_decision_changes'],1)

    def test_constant_native_family_can_be_preserved(self):
        r=audit((2.,2.,2.),2.)
        self.assertEqual(r['measured_mse'],0.)
        self.assertEqual(r['minimum_uniform_error'],0.)

    def test_missing_duplicate_and_unplanned_cases_reject(self):
        rows=[{'context':c['id'],'name':name} for c in contexts() for name in GPU_NAMES]
        validate_cohort(rows)
        for malformed in (rows[:-1],rows[:-1]+[rows[0]],rows+[rows[0]]):
            with self.assertRaises(ValueError):validate_cohort(malformed)
