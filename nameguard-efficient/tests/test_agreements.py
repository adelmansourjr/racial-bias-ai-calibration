import copy
import unittest
from nameguard.agreements import compare_reviews
from nameguard.contracts import ContractError
from test_annotations import proposal


class Agreements(unittest.TestCase):
    def test_identical_valid_reviews(self):
        p=proposal();r=compare_reviews(p,p)
        self.assertEqual(r['observed_character_agreement'],1.)
        self.assertEqual(r['cohen_kappa'],1.)
        self.assertEqual(r['identity_character_f1'],1.)
    def test_an_omission_is_reported_as_disagreement(self):
        p=proposal();q=copy.deepcopy(p);q['nodes'][1]['role']='evidence'
        r=compare_reviews(p,q)
        self.assertLess(r['cohen_kappa'],1.);self.assertLess(r['identity_character_f1'],1.)
    def test_different_sources_are_not_compared(self):
        p=proposal();q=copy.deepcopy(p);q['nodes'][1]['text']='Other'
        with self.assertRaises(ContractError):compare_reviews(p,q)
