# Subtractive LoRA: implemented, tested, and not a universal bias-removal result

> **Public source-and-aggregate release.** Learned adapter binaries, per-CV score
> rows, raw text and hidden-state caches are retained locally and are not uploaded.
> Demo and full-record verifier commands require those separately held local
> artifacts. This public checkout supports the mathematical tests, aggregate
> inspection and source-hash verification, not independent replay of private rows.


This is a real **internal rank-one residual activation adapter**. It receives
intact named inputs, runs the original frozen Qwen transformer, subtracts a learned
correction from the final normalized decision vector and uses the original
vocabulary head. No name detector, identity metadata, input redaction, anonymous
reference or second evidence stream is used at inference. Base weights and Clef
remain untouched.

**It does not meet the name-bias-removal goal.** Average score variation improves,
but decision flips remain and name-ID information is highly recoverable. These
negative findings are part of the result, not discarded unsuccessful settings.

Public checks (from the repository root):

```sh
python -m unittest discover -s subtractive-lora-pilot-v2/tests -v
python -m unittest discover -s subtractive-lora-score-aware/tests -v
python subtractive-lora-score-aware/scripts/verify_public.py
```

## Why subtraction is mathematically reasonable

For orthonormal name directions U, h'=h-UU^T(h-mu) removes their projection.
At an original linear head W this equals a subtractive rank-r LoRA update
W' = W-(WU)U^T, plus an offset when mu is nonzero. The subtraction is exact in
that specified subspace; it does not establish that all harmful name effects
occupy that subspace. Standard signed LoRA can already express negative updates.

The [first SVD pilot](../subtractive-lora-pilot-v2/README.md) found that its largest
name-variance direction did not reliably control score sensitivity. This follow-up
fits the score-relevant correction while penalizing shifts in different CV/job
contexts. It then uses

    h' = h - a [v^T(h-mu)],     a = w/(w^T w),

where w is the original A-minus-B head direction. The corrected margin is
w^T h - v^T(h-mu). This is a rank-one activation adapter, equivalent in exact
arithmetic to a low-rank head update with an affine offset. Its directions are
fitted by regularized least squares, not SGD/QLoRA. See [METHOD.md](METHOD.md).

## Actual confirmation results

Sixteen additional CVs, disjoint from the preceding 32-CV study, supply 32 job
contexts. Four fitting names and four additional names produce 256 intact-input
forwards. A fixed 64-item ARC-style regression adds 64 forwards. Selection uses
only eight cached fitting CVs and eight development CVs before these additional
CV outputs are observed. This is an adaptive exploratory follow-up, not external
preregistration or guaranteed disjointness from every historical local study.

| Measurement on new confirmation CVs | Native | Subtractive |
|---|---:|---:|
| Mean name-related A/B logit range | 0.72949 | 0.64800 |
| Mean name-related probability range | 0.12798 | 0.11347 |
| Contexts with name-dependent binary flips | 5/32 | 6/32 |
| Category-pair relevance diagnostic | 75.00% | 68.75% |
| Four-class name-ID probe accuracy | 96.88% | 96.88% |
| ARC-style regression, 64 items | 51/64 | 52/64 |

On the four additional names alone, mean probability range decreases from
0.09073 to 0.08165, but flip contexts increase from 5 to 6. Lower average range
does not prevent a small shift across the zero decision threshold. The probe is
a fixed ridge decoder with 25% chance, not a universal leakage estimator.

A descriptive CV-clustered bootstrap estimates a mean probability-range
change of -1.45 percentage points, with a 95% interval of -2.61 to -0.35 points.
This supports limited average-range reduction in this sample while threshold
flips and representation-leakage criteria still fail. It does not certify a
population fairness bound.

The category diagnostic is not human hiring accuracy. The 64 capability items
were used in earlier research and use our chat A/B/C/D readout; this is not a
standard full ARC benchmark or a formal noninferiority study. Two predictions
change; equal or better rounded accuracy cannot establish identical behavior.

Actual edited-head margins match the algebraic rank-one prediction to within
6.08e-6 in this FP32/quantized runtime. Exact matrix identities are not described
as bitwise equal GPU operations. The original checkpoint and parameter digest
verify unchanged after the run. Raw private activation caches and actual CV/job
text are not public artifacts.

## What the experiment establishes

1. A separate subtractive low-rank adapter can operate inside the scoring path
   with raw names intact and frozen base weights.
2. An SVD name projector can reduce some decoder accuracy without removing
   name-dependent scores. A score-aware correction can reduce average score
   variation while retaining highly decodable names and decision flips.
3. Neither final-state variant establishes universal name invariance, all-layer
   erasure, unchanged hiring utility or zero racial discrimination. Names here
   are controlled strings, not verified racial group labels.

All-layer subtractive training is a different experiment: early edits change
later activations, so a final-state fit cannot validate that mechanism. A next
study should train paired-name consistency with qualification/teacher retention,
use multiple prompts and scripts, re-extract directions after interventions,
and test a new untouched confirmation split. It must report failure if it cannot
retain utility while reducing both threshold effects and representation leakage.
The earlier learned all-block LoRA pilot also retained name information; merely
negating that adapter is not a mathematical inverse of its nonlinear effects.

## Use and reproduce

The actual learned file is `adapters/subtractive-rank-one.npz`, containing only
a, v and mu (three 1,536-element vectors). The frozen base is loaded separately.
With the tested Apple Silicon MLX environment:

```sh
python scripts/demo.py --checkpoint /path/to/exact/frozen-qwen \
  --prompt-file /path/to/intact-named-cv-and-question.txt
python scripts/verify.py
python -m unittest discover -s tests -v
```

The demo applies the selected adapter to every supplied prompt; it is not a
name-aware gate or a production hiring tool. The prompt still needs a suitable
A/B question. The basis and coefficient files are research artifacts, not
certified debiasing adapters.

`audit/plan.json` freezes the fitting mechanism, candidate grid and source hashes
before new confirmation inference. `audit/selection.json` records development
selection. `results/final.json` has all 256 score rows and 64 capability rows.
The source uses the cached private states from the preceding SVD pilot, so exact
refitting requires rerunning that pilot and separately obtaining the same data
and pinned checkpoint. The public verifier checks coverage, hashes and arithmetic;
it does not rerun the model or prove all-name fairness. Inference requires
NumPy and the pinned MLX stack from `../nameguard-validation/requirements-mlx.txt`.
Refitting requires the locally cached `nameguard-validation` contexts, Kaggle CV
file, cached job snapshot and benchmark items. The experiment refuses to
overwrite frozen plans: use a fresh sibling workspace copy without generated
audit/result outputs, preserve earlier studies, and rerun the SVD stage before
the score-aware stage.

Ten exact CPU arithmetic tests exercise cancellation, affine factorization,
task-direction conflict and the distinction between equal scores and retained
name states. The full source, unsuccessful first pilot and limitations remain
separate from the earlier structurally guaranteed name-channel system.

## Primary research context

Low-rank frozen-base updates: [Hu et al., LoRA](https://arxiv.org/abs/2106.09685v2).
Linear concept erasure with precise guardedness assumptions:
[Belrose et al., LEACE](https://arxiv.org/pdf/2306.03819v4).
Internal activation mitigation for hiring, including ablation utility risks:
[Karvonen and Marks](https://arxiv.org/html/2506.10922v1).
Our small mechanism pilot does not reproduce those papers' full experiments.

The released adapter is derived from real-excerpt activations. Excluding raw
text does not establish differential privacy or prevent all recovery of training
information; the learned file is not certified anonymized data.
