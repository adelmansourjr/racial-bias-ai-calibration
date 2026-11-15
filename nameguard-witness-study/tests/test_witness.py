import itertools
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_cases import cases, HELD_OUT, FIT, NAMES, DEV_NAMES, WITNESSES
from witness import LETTERS, checked, unchecked, choices, oracle_from_registry, prompt


class EvidenceWitnessTests(unittest.TestCase):
    def test_independent_design_oracle(self):
        for split in ("dev", "test"):
            for case in cases(split):
                with self.subTest(split=split, case=case.key):
                    actual = oracle_from_registry(case.target, case.entries)
                    self.assertEqual((actual.status, actual.record_ids),
                                     (case.expected_status, case.expected_ids))

    def test_any_possible_proposal_is_safe_and_incomplete_proposals_abstain(self):
        rejected_definitive = 0
        for case, proposal in itertools.product(cases("test"), (*LETTERS, "Z", None)):
            result = checked(case, proposal)
            if result.status == "review":
                if case.expected_status != "review":
                    rejected_definitive += 1
                continue
            self.assertEqual(result.status, case.expected_status)
            self.assertEqual(result.record_ids, case.expected_ids)
            self.assertIn(choices(case)[proposal], result.record_ids)
        self.assertGreater(rejected_definitive, 0)

    def test_checker_does_not_use_expected_labels(self):
        case = next(cases("test"))
        mislabeled = replace(case, expected_status="contradicted", expected_ids=("R4",))
        for letter in LETTERS:
            self.assertEqual(checked(case, letter), checked(mislabeled, letter))
        self.assertEqual(prompt(case), prompt(mislabeled))

    def test_checker_excludes_untrusted_proposal_and_requires_no_hidden_conflict(self):
        case = next(c for c in cases("test") if c.kind == "equal_conflict")
        preferred = next(letter for letter, rid in choices(case).items() if rid == "R1")
        self.assertEqual(unchecked(case, preferred).status, "confirmed")
        self.assertEqual(checked(case, preferred).status, "review")
        self.assertEqual(checked(case, preferred).record_ids, ("R1", "R2"))
        changed = replace(case, entries=tuple(replace(row, issuer="unverified") for row in case.entries))
        self.assertEqual(checked(changed, preferred).reason, "no_applicable_record")

    def test_names_and_credential_families_are_disjoint(self):
        self.assertEqual(len(set((*NAMES, *DEV_NAMES, *WITNESSES))), len(NAMES) + len(DEV_NAMES) + len(WITNESSES))
        self.assertTrue(set(FIT).isdisjoint(HELD_OUT))

    def test_name_invariance_does_not_require_same_output_after_fact_change(self):
        suite = tuple(c for c in cases("test") if c.family == 0)
        original = suite[0]
        for alternate in suite[1:]:
            choose_correct = lambda c: next((k for k, v in choices(c).items() if v in c.expected_ids), "E")
            self.assertNotEqual(checked(original, choose_correct(original)).serialize(),
                                checked(alternate, choose_correct(alternate)).serialize())
