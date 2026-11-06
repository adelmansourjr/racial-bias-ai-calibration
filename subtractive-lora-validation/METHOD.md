# Validation of exact output equality for subtractive LoRA

This study evaluates a frozen Qwen2.5-1.5B scorer with the original named prompt. It uses the next-token A/B margin and a rank-one activation edit. The design separates fitting, candidate selection, and final measurement.

## Design and controls

- Fit on 24 real CV excerpts, select on 8, and evaluate on 24 additional CVs. Each split has its own names. CV IDs and normalized exact excerpt text are disjoint across the three splits. Sampling excludes IDs and normalized excerpts from the earlier NameGuard validation and score-aware pilot. Disjointness from every other historical study and near-duplicate text was outside the check.
- Cross each test CV with one category-matched job, one category-mismatched job, and six names from the test pool. This yields 48 fixed CV/job contexts and 288 named input variants. Each name intervention holds the CV excerpt, job excerpt, and prompt style fixed. Both plain and selective prompt styles occur in each split. Names are string interventions, with no race labels.
- Fit 72 candidates from two activation feature sets, two context-score targets, three mean-retention weights, three ridge strengths, and two intervention strengths. The fit-only name-free teacher supplies one possible target. The selected adapter receives intact named text at inference. The fitting plan and source digests precede model calls, and the selected coefficients are hashed before test inference. This is a local freeze with no external registration.
- Use the native scorer and prior score-aware subtractive adapter as comparators, evaluated on exactly the same token sequences and final vocabulary head.
- Repeat eight selected identical inputs on the same runtime. Run the previous role-bound backend on those eight contexts with all six names and the known injected name spans. This positive control has a different input contract. The implementation compares digests for full arrays and compares the score records directly.
- Separately measure 48 paired synthetic qualification changes and a fixed 64-item ARC-style multiple-choice regression. These are discrete checks for decision collapse and capability changes. Category matching is a coarse relevance diagnostic. Neither it nor the synthetic rules provide independent human hiring labels. The ARC items were drawn from the existing project bank and use the project-specific choice readout.

The fixed acceptance criterion requires exact serialized score and FP32 margin equality for every test context, zero name-related flips, and no decrease in the category, rubric, or ARC accuracy point estimates. Utility point estimates on this small sample are screening gates. They do not support a formal noninferiority claim.

## Readout and learned edit

Let h be the final normalized state and w the original A-minus-B vocabulary direction. The learned correction reads h alone or the concatenation of normalized final-position states from layers 8, 16, 24, and 28. A ridge system learns the scalar correction c(x) from same-context contrasts and a penalty on context means. It is applied at the final state:

```text
a = w / (wᵀw)
h′ = h − a c(x)
margin′ = wᵀh − c(x)
```

This has rank one in the map from the chosen feature vector to the final-state update. Reading four layers supplies earlier features. The subtraction occurs once at the final state. If the features are h alone, it is equivalent in exact arithmetic to a rank-one affine update to the head. With earlier features, it is a cross-layer residual adapter and generally cannot be folded into a single ordinary LoRA weight matrix on h. The hardware check compares the edited-head margin against the algebraic identity.

For a name swap, write Δh for the change in final state and Δc for the change in learned correction. The margin is invariant exactly when wᵀΔh = Δc. Exact state equality also requires Δh = aΔc. Any component of Δh orthogonal to a survives this edit. Full vocabulary equality requires W(Δh − aΔc) = 0, which is stronger than one A/B margin equality. A repeatable score on this test does not establish invariance for arbitrary raw text or unbiased hiring.

## Equality and uncertainty

- Exact FP32 margin compares binary representations. Exact serialized score includes unrounded margin, conditional A probability and A/B decision. A/B logits, full next-token vocabulary logits and final activations are compared separately via shape/dtype/value digests.
- The engineering tolerance is 0.10 A-minus-B logit units. The logistic derivative is at most 1/4, so this bounds the conditional A-probability range by 2.5 percentage points. It gives no protection against threshold crossing near zero. The gate counts flips explicitly.
- Use CV clusters for paired resampling. The supplemental analysis resamples within the eight fixed occupation strata. Repeated names and jobs for one CV remain together. These intervals describe resampling variation in this dataset, with convenience sampling and a fixed name pool.
- The shortlist analysis considers independent name assignments from the six-name pool to the six candidates for each job. Tie breaking uses stable opaque CV IDs. A candidate's best rank occurs at its maximum score with all other candidates at their minima. Its worst rank occurs at its minimum with all others at their maxima. These extrema certify possible and guaranteed top-three membership over the finite assignments.
- A separate retrospective geometry diagnostic uses earlier cached states. It tests full fit-span projection and an exact-to-numerical-precision least-squares fit of calibration score constraints. Its results are independent of candidate selection in this study.

This protocol adapts the idea of separately testing invariance and directional behavior from [Ribeiro et al., CheckList](https://arxiv.org/abs/2005.04118). [Karvonen and Marks](https://arxiv.org/html/2506.10922v1) provide a distinct precedent for internal affine interventions in hiring evaluations. The rank-one implementation uses the low-rank factorization described in [Hu et al., LoRA](https://arxiv.org/abs/2106.09685).

## Audit corrections and study ordering

The follow-up role-bound control was expanded after the primary run, using the
same 48 contexts and six names. This is a confirmatory implementation check of an
existing structural contract, separate from learned adapter selection. It compares
array bytes directly at the final hidden and full-vocabulary boundaries. A native
forward on the retained evidence supplies a separate reference for every context.

The frozen runner's test summary used each arm as its own reference for the
`mean_context_drift` field. The independent audit reconstructs this field against
the native scorer and writes `results/verified-summary.json`. The original
summary stays available as local provenance. Development selection used the
correct reference. No test acceptance gate used the faulty field.

The original CV and job text comes from the project's local snapshots of the
[Kaggle résumé dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
and the [DataStax LinkedIn job listings dataset](https://huggingface.co/datasets/datastax/linkedin_job_listings).
