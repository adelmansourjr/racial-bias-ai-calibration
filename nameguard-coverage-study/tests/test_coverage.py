from collections import Counter
from dataclasses import replace
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_cases import cases, DEV_CREDENTIALS, TEST_CREDENTIALS
from coverage_policy import review, proposals


class VerificationTests(unittest.TestCase):
    def test_decisive_control_positions_balanced_and_families_disjoint(self):
        self.assertTrue(set(DEV_CREDENTIALS).isdisjoint(TEST_CREDENTIALS))
        positions = Counter(i for c in cases("test") if len(c.expected_ids) == 1
                            for i, r in enumerate(c.entries) if r.record_id in c.expected_ids)
        self.assertEqual(positions, {0: 12, 1: 12, 2: 12, 3: 12})

    def test_one_proposal_covers_every_ordering_of_model_scores(self):
        for case, scores in itertools.product(cases("test"), itertools.permutations(range(5))):
            outcome, count = review(case, scores, "latest_top_one")
            self.assertEqual((outcome.status, outcome.record_ids), (case.expected_status, case.expected_ids))
            self.assertEqual(count, 0 if outcome.status == "review" else 1)

    def test_stale_applicable_proposals_raise_a_real_coverage_failure(self):
        case = next(cases("test"))
        stale_indices = [i for i, r in enumerate(case.entries)
                         if r.record_id not in case.expected_ids and r.holder == "applicant"]
        scores = [10. if i in stale_indices else -10. for i in range(5)]
        old, _ = review(case, scores, "applicable_top_two")
        improved, _ = review(case, scores, "latest_top_one")
        self.assertEqual(old.status, "review")
        self.assertEqual(improved.status, "confirmed")

    def test_source_authority_is_part_of_the_policy(self):
        c = next(cases("test"))
        untrusted = replace(c, entries=tuple(replace(r, issuer="unverified") for r in c.entries))
        report, count = review(untrusted, (4, 3, 2, 1, 0), "latest_top_one")
        self.assertEqual((report.status, report.record_ids, count), ("review", (), 0))

    def test_deterministic_report_ignores_storage_order(self):
        case = next(cases("test"))
        a, _ = review(case, (0, 1, 2, 3, 4), "latest_top_one")
        b, _ = review(replace(case, entries=case.entries[::-1]), (0, 1, 2, 3, 4), "latest_top_one")
        self.assertEqual(a.serialize(), b.serialize())
        self.assertEqual(review(case, None, "latest_top_one"), (None, 0))

    def test_ties_at_the_latest_date_keep_all_controlling_ids(self):
        case = next(cases("test"))
        current = next(row for row in case.entries if row.record_id in case.expected_ids)
        duplicate_finding = replace(current, record_id="TIE")
        other = tuple(row for row in case.entries if row.record_id != current.record_id)[:2]
        case = replace(case, entries=(current, duplicate_finding, *other))
        first, first_count = review(case, (2, 1, 0, 0, 0), "latest_top_one")
        second, second_count = review(case, (1, 2, 0, 0, 0), "latest_top_one")
        self.assertEqual(first.serialize(), second.serialize())
        self.assertEqual(first.record_ids, tuple(sorted((current.record_id, "TIE"))))
        self.assertEqual((first_count, second_count), (1, 1))
