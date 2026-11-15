"""Selective evidence scoring and exact finite-population audit bounds."""
from fractions import Fraction
from math import comb, isfinite, nan

MODES = ("single", "mean", "minimum")


def score(pairs, mode):
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Expected one A/B logit pair per mapping")
    if any(not isfinite(value) for pair in pairs for value in pair):
        raise ValueError("Logits must be finite")
    a = pairs[0][0] - pairs[0][1]
    b = pairs[1][1] - pairs[1][0]
    return {"single": a, "mean": (a + b) / 2, "minimum": min(a, b)}[mode]


def ranking(rows, mode):
    return sorted(((score(row["logits"], mode), key) for key, row in rows.items()
                   if row["admitted"]), reverse=True)


def choose(rows, labels, mode, max_rate=Fraction(1, 25), min_accepts=12):
    ranked = ranking(rows, mode)
    threshold = None
    errors = accepted = 0
    for index, (value, key) in enumerate(ranked, 1):
        errors += not labels[key]
        last_tie = index == len(ranked) or ranked[index][0] != value
        if last_tie and index >= min_accepts and Fraction(errors, index) <= max_rate:
            threshold, accepted = value, index
    predictions = decisions(rows, mode, threshold)
    return dict(mode=mode, threshold=threshold, accepted=accepted,
                errors=sum(not labels[key] for key in predictions))


def decisions(rows, mode, threshold):
    return sorted(key for value, key in ranking(rows, mode)
                  if threshold is not None and value >= threshold)


def sample(accepted, audit_order, maximum):
    size = min(maximum, (4 * len(accepted) + 4) // 5)
    accepted_set = set(accepted)
    if len(set(audit_order)) != len(audit_order) or not accepted_set <= set(audit_order):
        raise ValueError("Audit order must be a permutation covering the accepted set")
    return sorted([key for key in audit_order if key in accepted_set][:size])


def tail_numerator(total, errors, sample_size, observed):
    low = max(0, sample_size - (total - errors))
    high = min(observed, errors, sample_size)
    return sum(comb(errors, draw) * comb(total - errors, sample_size - draw)
               for draw in range(low, high + 1))


def upper_error_count(total, sample_size, observed, alpha=Fraction(1, 20)):
    if not (0 <= observed <= sample_size <= total and 0 < alpha < 1):
        raise ValueError("Invalid audit counts or alpha")
    denominator = comb(total, sample_size)
    low, high = observed, total - sample_size + observed
    while low < high:
        mid = (low + high + 1) // 2
        tail = tail_numerator(total, mid, sample_size, observed)
        if tail * alpha.denominator > denominator * alpha.numerator:
            low = mid
        else:
            high = mid - 1
    return low


def auc(rows, labels, mode):
    values = ranking(rows, mode)
    pos = [v for v, key in values if labels[key]]
    neg = [v for v, key in values if not labels[key]]
    if not pos or not neg:
        return nan
    return sum((p > n) + (p == n) / 2 for p in pos for n in neg) / (len(pos) * len(neg))
