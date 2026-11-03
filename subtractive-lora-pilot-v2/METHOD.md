# Subtractive low-rank adapters: what cancels, what remains

The base weights are frozen. An activation is the temporary vector computed for
an input, not a separate permanent model parameter. This experiment learns a
small cancellation operator and attaches it after the final normalized decision
state. Raw named inputs and all earlier transformer computation remain intact.
This is a final-state/output-head intervention, not all-layer scrubbing.

## Exact low-rank construction

For column-vector hidden state h, vocabulary head W and orthonormal direction
matrix U (d by r), define

    h' = h - alpha U U^T (h - mu).
    z' = W h - alpha (W U) U^T h + alpha (W U) U^T mu.

Thus A=U^T and B=-alpha WU give a rank-at-most-r subtractive LoRA update BA
to W. When mu=0 this is an ordinary linear low-rank head update. When mu is
nonzero an explicit affine bias term is required (or augment h with a constant
coordinate). The centered variant must not be called an ordinary bias-free LoRA.
The implementation saves both representations and checks their numerical
agreement on actual frozen-model states. Separate FP32 evaluation paths can
round differently, so algebraic equality is not asserted as bitwise equality.

The small runtime implementation applies h' before the original vocabulary head,
avoiding a very large vocabulary-sized stored B factor. That is a residual
activation adapter equivalent to this head update in exact arithmetic. It is
not merged into or written over the original model files.

## What the minus sign does not add

Ordinary LoRA learns W+BA with unrestricted signed factors. Every W-BA is already
representable as W+B(-A). Simply reversing a fitted adapter's sign adds no
expressive power and is not the inverse of a nonlinear multilayer computation.
Subtractive structure becomes useful by constraining BA to a justified
cancellation operator, rather than by choosing a negative scale alone.

## Identifying directions without confusing qualifications

For each fitting CV/job context q, collect h(q,n) under four name interventions.
Let h_bar(q) be the mean across those names. Stack the paired differences
h(q,n)-h_bar(q) into D and use its right singular vectors to obtain candidate U.
Holding the text and job fixed helps isolate the header intervention. It does
not prove that the resulting subspace has no useful overlapping information.

We use eight fitting CVs, eight development CVs and sixteen CVs for the held-out
pilot. Ranks, alpha values and centering are selected only on development
name-range and simple retention criteria specified before inference. Four
additional evaluation names are absent from this adapter's fitting/development.
The CVs were used in earlier studies; this is an exploratory mechanism pilot,
not a newly untouched real-world confirmation sample.

## Exact cancellation condition and its limit

At alpha=1, U^T(h'-mu)=0 in exact arithmetic. For two names at fixed q,

    h'(q,n)-h'(q,n') = (I-UU^T)[h(q,n)-h(q,n')].

The full edited state is name-invariant if and only if every relevant name
difference lies in col(U). A finite fitting sample cannot establish this for
every new name, CV and prompt. A linear A/B margin w^T h' can be invariant under
the weaker condition w^T(I-UU^T)delta=0, while names remain decodable elsewhere.

For that margin,

    |delta_margin| <= ||w||_2 ||(I-UU^T)delta||_2.

For a unit qualification direction v, its change is bounded by
||U^T v||_2 ||U^T(h-mu)||_2. Complete preservation of every vector in a task
subspace V requires U^T V=0. If an indispensable qualification signal shares a
direction with the targeted name difference, removing that direction while
preserving it is mathematically incompatible. Empirical retention must be
measured, not promised from low rank alone.

Alpha below one shrinks rather than annihilates components. Nonlinear name
information can remain outside U, and a single final-state adapter cannot erase
earlier activations. Multi-layer adapters require a new evaluation because
downstream states and nonlinear interactions change after an early edit.

## Primary sources

Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, describe
frozen-base low-rank updates: https://arxiv.org/abs/2106.09685v2 .
Belrose et al. (2023; manuscript v4, 2025), *LEACE*, give a more specific
convex-loss linear guardedness criterion, distinct from universal independence:
https://arxiv.org/pdf/2306.03819v4 .
Karvonen and Marks (2025), Section 3.4, use internal affine concept editing for
hiring fairness and document utility risks of naive ablation:
https://arxiv.org/html/2506.10922v1 .

This SVD pilot is not a reproduction of LEACE or the latter paper's all-layer
ACE experiment. It supplies a controlled subtractive baseline with its own
split, validation rules and negative outcomes preserved.
