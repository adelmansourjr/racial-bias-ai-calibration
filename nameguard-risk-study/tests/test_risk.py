from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from math import comb
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk import choose, decisions, sample, score, tail_numerator, upper_error_count
from run import evidence_bytes


class TestAudit(unittest.TestCase):
    def test_small_populations_exact_coverage(self):
        alpha = Fraction(1, 20)
        for total in range(1, 16):
            for size in range(total + 1):
                for errors in range(total + 1):
                    probability_of_undercoverage = Fraction(0)
                    for x in range(max(0, size - (total - errors)), min(size, errors) + 1):
                        if upper_error_count(total, size, x) < errors:
                            probability_of_undercoverage += Fraction(
                                comb(errors, x) * comb(total - errors, size - x), comb(total, size))
                    self.assertLessEqual(probability_of_undercoverage, alpha)

    def test_tail_against_enumeration(self):
        for N in range(1, 8):
            for K in range(N + 1):
                for m in range(N + 1):
                    draws = list(combinations(range(N), m))
                    for x in range(m + 1):
                        self.assertEqual(tail_numerator(N, K, m, x),
                                         sum(sum(i < K for i in draw) <= x for draw in draws))

    def test_restricting_random_order_is_uniform(self):
        accepted = list("abcde")
        draws = Counter(tuple(sample(accepted, p, 160)) for p in permutations("abcdef"))
        self.assertEqual(len(draws), 5)
        self.assertEqual(len(set(draws.values())), 1)

    def test_edge_cases(self):
        self.assertEqual(upper_error_count(40, 40, 3), 3)
        self.assertEqual(upper_error_count(40, 0, 0), 40)
        self.assertEqual(upper_error_count(40, 32, 0), 1)
        with self.assertRaises(ValueError):
            upper_error_count(3, 4, 1)


class TestSelectiveDecision(unittest.TestCase):
    def test_margins(self):
        self.assertEqual(score([[6, 4], [0, 6]], "mean"), 4)
        self.assertEqual(score([[6, 4], [0, 6]], "minimum"), 2)

    def test_keeps_ties_in_same_decision(self):
        rows = {k: {"admitted": True, "logits": [[v, 0], [0, v]]}
                for k, v in (("a", 2), ("b", 1), ("c", 1))}
        labels = {"a": True, "b": True, "c": False}
        rule = choose(rows, labels, "mean", Fraction(1, 4), 1)
        self.assertEqual(decisions(rows, "mean", rule["threshold"]), ["a"])

    def test_report_offsets_validate_unicode(self):
        item = {"answer": "ó", "start": 1, "passage": "Có"}
        self.assertIn("ó".encode(), evidence_bytes(item, True))
        with self.assertRaises(ValueError):
            evidence_bytes(dict(item, start=0), True)


if __name__ == "__main__":
    unittest.main()
