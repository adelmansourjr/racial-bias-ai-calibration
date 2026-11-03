"""Exact finite counterexamples for overbroad research claims; not model trials."""
from fractions import Fraction
import json


def behavioral_equality_is_not_erasure():
    states = [(7, name) for name in ('John', 'Mohammed')]
    return {'equal_scores': len({h[0] for h in states}) == 1,
            'names_still_recoverable': [h[1] for h in states] == ['John', 'Mohammed']}


def zero_covariance_is_not_independence():
    rows = [(-1, 0), (1, 0), (0, 1), (0, 1)]
    mean = lambda a: sum(map(Fraction, a)) / len(a)
    mx, mz = mean([x for x, z in rows]), mean([z for x, z in rows])
    covariance = mean([x*z for x, z in rows]) - mx*mz
    # Predict class 0 only for x > 0: 3 of 4 correct despite equal centroids.
    acc = mean([int((0 if x > 0 else 1) == z) for x, z in rows])
    nonlinear = mean([int((0 if abs(x) == 1 else 1) == z) for x, z in rows])
    return {'covariance': str(covariance), 'threshold_accuracy': str(acc),
            'nonlinear_accuracy': str(nonlinear)}


def finite_test_is_not_universal():
    observed = ['John', 'Mohammed', 'Maria']
    score = lambda name: 1 if name == 'UNSEEN' else 0
    return {'observed_equal': len({score(n) for n in observed}) == 1,
            'unseen_violation': score('UNSEEN') != score(observed[0])}


def agreement_is_not_truth():
    shared_role_payload = ('John',)  # Both reviewers omit the alias Leaf.
    oracle_identity = {'John', 'Leaf'}
    return {'two_reviews_agree': shared_role_payload == tuple(['John']),
            'identity_coverage_complete': set(shared_role_payload) == oracle_identity}


def canonical_baseline_equivalence():
    # Algebraic witness, not a Qwen latency or quality measurement.
    documents = [(('identity', n), ('evidence', 'Python SQL'))
                 for n in ('John', 'Mohammed', 'unseen')]
    q = lambda d: ''.join(text for role, text in d if role == 'evidence')
    tokenizer = lambda text: tuple(text.encode('utf-8'))
    scorer = lambda tokens: sum(tokens)
    external = [scorer(tokenizer(q(d))) for d in documents]
    internal = []
    for d in documents:
        raw_private = tokenizer(''.join(text for role, text in d))
        assert raw_private  # Raw visit is not needed by the score computation.
        internal.append(scorer(tokenizer(q(d))))
    return {'same_outputs': internal == external,
            'no_internal_placement_fairness_advantage_proved': True}


def report():
    return {'kind': 'exact finite mathematical witnesses, not new GPU experiments',
            'behavior_vs_encoding': behavioral_equality_is_not_erasure(),
            'linear_vs_independence': zero_covariance_is_not_independence(),
            'finite_vs_universal': finite_test_is_not_universal(),
            'agreement_vs_truth': agreement_is_not_truth(),
            'external_vs_internal': canonical_baseline_equivalence()}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
