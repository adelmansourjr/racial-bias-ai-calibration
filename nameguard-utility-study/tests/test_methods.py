from collections import Counter
from datetime import date, timedelta
import itertools
import re
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import scenarios, DEV, TEST
from methods import ROTATIONS, STATUSES, cycle_scores, paired_margin, classify, decision_report
from compositional_fixtures import challenge, NEW_FAMILIES
from evidence_policy import Observation, evaluate


class MathematicalContract(unittest.TestCase):
    def test_two_way_removes_shared_additive_margin(self):
        for semantic, offset in itertools.product(range(-8, 9), range(-8, 9)):
            self.assertEqual(paired_margin(semantic + offset, -semantic + offset), semantic)

    def test_cyclic_removes_label_bias_from_status_differences(self):
        signal = dict(zip(STATUSES, (2, 9, -1), strict=True))
        letter_bias = (10, -3, 5)
        scores = cycle_scores([tuple(signal[name] + b + j for name, b in zip(mapping, letter_bias, strict=True))
                              for j, mapping in enumerate(ROTATIONS)])
        self.assertEqual(classify(scores), "contradicted")
        self.assertAlmostEqual(scores[1] - scores[0], 7)
        self.assertAlmostEqual(scores[0] - scores[2], 3)
        self.assertEqual(classify((0, 0, -1)), "review")

    def test_report_is_byte_stable_and_status_specific(self):
        self.assertEqual(decision_report("credential", "confirmed").as_bytes(),
                         decision_report("credential", "confirmed").as_bytes())
        self.assertNotEqual(decision_report("credential", "confirmed").as_bytes(),
                            decision_report("credential", "contradicted").as_bytes())

    def test_studies_use_disjoint_families_and_crossed_subject_time_bags(self):
        self.assertEqual(len({task.credential for task in (*DEV, *TEST, *NEW_FAMILIES)}), 42)
        for split, n in (("dev", 36), ("test", 144)):
            self.assertEqual(len(list(scenarios(split))), n)
        cases = {case.key: case for case in challenge("test")}
        for family in range(12):
            for kind in ("subject", "time"):
                left, right = (cases[f"{family}-{kind}_{i}"] for i in (0, 1))
                self.assertNotEqual(left.status, right.status)
                bag = lambda cv: Counter(re.findall(r"\b\w+\b", cv.casefold()))
                self.assertEqual(bag(left.cv), bag(right.cv))


class ReviewedEvidencePolicy(unittest.TestCase):
    def test_subject_time_relevance_conflict_missing_and_order(self):
        today = date(2026, 11, 16)
        yesterday = today - timedelta(days=1)
        valid = Observation("x", "applicant", "A", today, "valid")
        invalid = Observation("y", "applicant", "A", today, "invalid")
        irrelevant = (Observation("third", "third_party", "A", today, "valid"),
                      Observation("old", "applicant", "A", yesterday, "invalid"),
                      Observation("future", "applicant", "A", today + timedelta(days=1), "invalid"),
                      Observation("other", "applicant", "B", today, "invalid"))
        for actual, expected in (((), "review"), ((valid,), "confirmed"), ((invalid,), "contradicted"),
                                 ((valid, invalid), "review")):
            self.assertEqual(evaluate("A", today, actual).status, expected)
            records = actual + (irrelevant if actual else tuple(o for o in irrelevant if o.record_id != "old"))
            reference = evaluate("A", today, records)
            self.assertEqual(reference.status, expected)
            for perm in itertools.permutations(records):
                self.assertEqual(evaluate("A", today, perm).serialize(), reference.serialize())

    def test_conflicting_duplicate_identifier_rejects(self):
        today = date(2026, 11, 16)
        with self.assertRaises(ValueError):
            evaluate("A", today, (Observation("x", "applicant", "A", today, "valid"),
                                  Observation("x", "applicant", "A", today, "invalid")))
