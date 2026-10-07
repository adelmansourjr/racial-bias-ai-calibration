"""Synthetic checks of scoped fairness proofs and counterexamples.

No real resumes, model weights, or hiring outcomes are used.
Run with Python + NumPy. Results are demonstrations, not a model certificate.
"""

from pathlib import Path
import json
import math
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def invariant_projector(differences):
    """Orthogonal projection annihilating the supplied paired differences."""
    d = np.asarray(differences, dtype=float)
    u, s, _ = np.linalg.svd(d.T, full_matrices=False)
    tolerance = max(d.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
    basis = u[:, s > tolerance]
    return np.eye(d.shape[1]) - basis @ basis.T, basis.shape[1]


def main():
    findings = {"data_kind": "synthetic mathematical demonstrations"}

    # Opposite discriminatory treatment cancels in the aggregate.
    decisions_a = np.array([1, 0])
    decisions_b = np.array([0, 1])
    gap = abs(decisions_a.mean() - decisions_b.mean())
    flips = np.mean(decisions_a != decisions_b)
    assert gap == 0 and flips == 1
    findings["aggregate_parity_counterexample"] = {
        "shortlisting_rate_gap": float(gap), "paired_flip_rate": float(flips)
    }

    # Zero cosine can represent a rotated, fully preserved identity signal.
    old_direction = np.array([1.0, 0.0])
    new_direction = np.array([0.0, 1.0])
    cosine = old_direction @ new_direction
    labels = np.array([-1, 1])
    hidden = labels[:, None] * new_direction
    accuracy = np.mean(np.sign(hidden[:, 1]) == labels)
    assert cosine == 0 and accuracy == 1
    findings["rotation_counterexample"] = {
        "old_new_cosine": float(cosine), "identity_recovery_accuracy": float(accuracy)
    }

    # Equal means / zero covariance do not imply nonlinear erasure, or even
    # absence of every profitable affine threshold under 0-1 loss.
    h = np.array([-1.0, 1.0, -2.0, 2.0])
    a = np.array([0, 0, 1, 1])
    means = [float(h[a == k].mean()) for k in [0, 1]]
    affine_accuracy = np.mean((h > 1.5) == a)
    nonlinear_accuracy = np.mean((np.abs(h) > 1.5) == a)
    assert means == [0, 0] and affine_accuracy == .75 and nonlinear_accuracy == 1
    findings["zero_mean_counterexample"] = {
        "class_conditional_means": means,
        "affine_threshold_accuracy": float(affine_accuracy),
        "nonlinear_probe_accuracy": float(nonlinear_accuracy)
    }

    rng = np.random.default_rng(180)
    q = rng.normal(size=(1000, 2))
    identity_variants = np.array([-1.0, 1.0])

    def state(qualifications, identity):
        identity = np.broadcast_to(identity, len(qualifications))
        return np.column_stack([
            qualifications, identity,
            identity * qualifications[:, 0], identity * qualifications[:, 1]
        ])

    weights = np.array([1.7, .5, .8, .4, .7])

    def score(hidden_states):
        # Nonlinear output, including an identity/task interaction.
        return sigmoid(hidden_states @ weights + .2 * hidden_states[:, 2] ** 3)

    orbit_scores = np.column_stack([score(state(q, identity))
                                    for identity in identity_variants])
    original_score_range = np.max(np.ptp(orbit_scores, axis=1))
    def corrected_score(received_state):
        # In this toy, extraction is provably just reading the same first
        # two qualification coordinates. Real extraction needs its own audit.
        recovered_q = received_state[:, :2]
        scores = np.column_stack([score(state(recovered_q, identity))
                                  for identity in identity_variants])
        return scores.mean(axis=1)

    corrected_for_a = corrected_score(state(q, -1))
    corrected_for_b = corrected_score(state(q, 1))
    orbit_residual = np.max(np.abs(corrected_for_a - corrected_for_b))
    assert original_score_range > .1 and orbit_residual == 0
    findings["fixed_orbit_average"] = {
        "max_original_score_gap": float(original_score_range),
        "max_corrected_score_gap": float(orbit_residual),
        "scope": "fixed identity interventions with qualifications preserved"
    }

    # Fit differences on a finite grid. The structural formula guarantees
    # that its span covers every possible identity difference in this toy.
    fit_q = np.array([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
    differences = state(fit_q, 1) - state(fit_q, -1)
    projector, rank = invariant_projector(differences)
    projected_a = state(q, -1) @ projector.T
    projected_b = state(q, 1) @ projector.T
    residual = np.max(np.abs(projected_a - projected_b))
    score_residual = np.max(np.abs(score(projected_a) - score(projected_b)))
    utility_target = (q @ weights[:2]) >= 0
    corrected_accuracy = np.mean((score(projected_a) >= .5) == utility_target)
    assert rank == 3 and residual < 1e-12 and score_residual < 1e-12
    assert np.max(np.abs(projected_a[:, :2] - q)) < 1e-12
    assert corrected_accuracy == 1
    findings["projection_with_structural_span_assumption"] = {
        "bias_span_rank": rank, "representation_dimension": 5,
        "max_held_out_hidden_gap": float(residual),
        "max_held_out_score_gap": float(score_residual),
        "toy_qualification_accuracy": float(corrected_accuracy),
        "scope": "all differences in this explicit synthetic structural model"
    }

    # An unseen identity direction violates the span assumption.
    p, _ = invariant_projector(np.array([[0.0, 2.0, 0.0]]))
    held_out_difference = np.array([0.0, 0.0, 2.0])
    unseen_residual = np.linalg.norm(p @ held_out_difference)
    assert unseen_residual == 2
    findings["unseen_direction_failure"] = {
        "held_out_hidden_gap": float(unseen_residual),
        "lesson": "training-span erasure does not prove unseen-span erasure"
    }

    p_full, rank_full = invariant_projector(np.eye(2))
    assert rank_full == 2 and np.max(np.abs(p_full)) < 1e-12
    findings["utility_collapse_counterexample"] = {
        "bias_span_rank": rank_full, "dimension": 2,
        "projected_states": (np.eye(2) @ p_full.T).tolist()
    }

    # Exact one-sided binomial bound after zero violations, fixed auditor,
    # independent candidate clusters. Multiple metrics use alpha/M.
    alpha, n, metrics = .05, 3000, 1
    bound = -math.expm1(math.log(alpha / metrics) / n)
    findings["zero_failure_bound"] = {
        "independent_candidates": n, "violations": 0,
        "simultaneous_confidence": 1 - alpha, "number_of_metrics": metrics,
        "population_violation_rate_upper_bound": bound,
        "required_n_for_1_percent_M_equals_1": math.ceil(math.log(alpha) / math.log(.99))
    }
    assert .00099 < bound < .00101

    # An insensitive judge can attenuate a real disparity.
    sensitivity, specificity = .6, .6
    true_rates = np.array([.2, .8])
    observed_rates = (sensitivity + specificity - 1) * true_rates + 1 - specificity
    assert np.isclose(np.diff(observed_rates)[0], .12)
    findings["judge_measurement_error"] = {
        "true_disparity": float(np.diff(true_rates)[0]),
        "observed_disparity": float(np.diff(observed_rates)[0]),
        "assumption": "same sensitivity/specificity across groups"
    }

    # Enumerate a tiny autoregressive vocabulary over four steps. The
    # fixed-bank token mixture yields the same entire sequence distribution.
    def sequence_distribution(received_identity, corrected):
        distribution = {(): 1.0}
        for t in range(4):
            updated = {}
            for history, probability in distribution.items():
                def next_probability(identity):
                    return float(sigmoid(.3 * sum(history) + .4 * t + .9 * identity))
                p1 = (sum(next_probability(identity) for identity in identity_variants) / 2
                      if corrected else next_probability(received_identity))
                updated[history + (0,)] = probability * (1 - p1)
                updated[history + (1,)] = probability * p1
            distribution = updated
        return np.array([distribution[key] for key in sorted(distribution)])

    original_a = sequence_distribution(-1, False)
    original_b = sequence_distribution(1, False)
    repaired_a = sequence_distribution(-1, True)
    repaired_b = sequence_distribution(1, True)
    original_tv = np.abs(original_a - original_b).sum() / 2
    repaired_tv = np.abs(repaired_a - repaired_b).sum() / 2
    assert original_tv > .1 and repaired_tv == 0
    assert np.isclose(repaired_a.sum(), 1)
    findings["autoregressive_token_mixture"] = {
        "number_of_enumerated_sequences": 16,
        "original_sequence_total_variation": float(original_tv),
        "corrected_sequence_total_variation": float(repaired_tv),
        "scope": "toy four-token sequences; general product-kernel proof in report"
    }

    output = Path(__file__).with_name("synthetic-verification-results.json")
    output.write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps(findings, indent=2))
    print("All mathematical demonstration assertions passed.")


if __name__ == "__main__":
    main()
