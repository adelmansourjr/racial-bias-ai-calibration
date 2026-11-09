# Automated scientific-method review B

Reviewer: AI agent, reviewer B. Date: 2026-11-09.

This is a separate automated technical review after the blinded annotation submission was sealed. It is not independent human validation, external replication, or a literature-verification exercise. I did not edit the sealed annotations, manuscript, code, or results, and did not inspect reviewer A.

## Materials and limits

This phase read only `paper/manuscript.md`, `gpu_study.py`, `boundary_study.py`, `analyze.py`, `results/summary.json`, `results/gpu/plan.json`, and `review-packet/PROTOCOL.md`. Consequently I cannot verify the imported implementations, checkpoint reconstruction recipe, raw result rows, timing follow-up, ARC retrieval, or cited literature. A missing item from this restricted set is a verification limit, not proof that it is absent from the repository.

The current `gpu_study.py` SHA-256 matches `plan.json` (`eb2bd1bf8420fadd23813f89226a0017697bcad8e2611207065f94a73246133b`); the current `analyze.py` SHA-256 matches `summary.json` (`22474c0e1914f0be019fc9f33c2b4f60631d69827c93655b786106c3c29a7e15`). I did not rerun GPU inference or analyses.

The manuscript already states many crucial qualifications correctly: one model/runtime, constructed names rather than racial populations, eight task families rather than 192 independent applicants, descriptive intervals, no noninferiority conclusion, exact same-ID external equivalence, utility loss, and no independent human validation. Those statements should remain. The principal unresolved issues follow.

## 1. High — contract selection and witness search are confounded in the anchor comparison

**Issue.** The claimed 158-to-170 admission improvement compares two changes together: selecting a different contract and searching additional feasible witnesses at admission. It does not isolate the benefit of maximum-support contract selection over a fixed anchor with the same runtime matching policy.

**Evidence.** `boundary_study.py:32-33` selects the anchor from the first fitting member's first candidate. At test time, lines 41 and 45 require the test member's *first* candidate to equal that anchor. In contrast, line 42 searches *any* candidate for the compiled contract. `gpu_study.py:122-134,138` repeats this asymmetry. Manuscript section 5.1 reports the 12-case gain as greedy-anchor versus compiler performance. That is a valid comparison with this particular greedy pipeline, but not a clean attribution to contract selection.

**Correction.** Preserve these frozen results and add a clearly labeled ablation: fixed fitting anchor plus exhaustive exact-contract witness matching, then compare that with the compiled contract using the identical matching rule. Also retain first-witness greedy as a separate baseline. Report paired admission changes by layout/name for each transition. Until then, describe the gain as the combined compiler-and-matching pipeline gain over the specified greedy pipeline.

## 2. High — the published initial timing interval uses dependent case units

**Issue.** The timing summary bootstraps eight named cases as if they supplied eight independently sampled timing units. The source collects those eight cases from at most two CV conditions, with multiple names sharing evidence and execution behavior. This creates pseudoreplication for a general speed-comparison interval; five repeats do not supply new task families either.

**Evidence.** `gpu_study.py:115,125,166-167` fills `timing_cases` with the first eight accepted requests. Six names are declared in `plan.json`, and all 192 requests are accepted in `summary.json`, so the first six timing cases come from one condition and the next two from the second. `analyze.py:37-40,55` reduces repeats to a median per case and bootstraps the eight ratios. The summary reports mean ratio 1.01525, interval [1.00448,1.02404], and `units: 8`. These are eight named timing cases, not eight task families. The manuscript explicitly acknowledges the initial imbalance and mentions a balanced follow-up; that acknowledgment is appropriate, but this summary is still the initial result.

**Correction.** Label the initial ratio and interval as a descriptive resampling of eight dependent named cases, with the context/family coverage shown explicitly. Do not use its interval to claim a general statistically resolved overhead. Give the balanced follow-up its own source, plan, result provenance, and analysis, with task-family/session units and paired repeat handling specified. I cannot assess that follow-up within the permitted materials.

## 3. Medium — paired timing is analyzed as a ratio of separate medians

**Issue.** Randomized paired repetitions are collected, but their pairing is discarded when calculating the endpoint. Under concurrent load and time drift, a ratio of arm-specific medians is not the same estimand as a paired repeat ratio or paired elapsed-time difference.

**Evidence.** `gpu_study.py:174-185` randomizes arm order within each repeat. `analyze.py:38-40` groups only by case and arm, then divides two marginal medians. The manuscript section 4.3 states that other training jobs remain active. No session-level replication, load trace, or thermal/energy state is visible in the permitted result summary.

**Correction.** Keep the existing endpoint frozen, but publish repeat-level paired differences/ratios and the randomized order as an additional descriptive analysis. For a stronger efficiency claim, plan multiple balanced sessions, record host/load conditions, and aggregate at a defensible task/session unit. Separately report setup and end-to-end costs if discussing deployment efficiency; current score-only timings deliberately omit contract compilation and semantic review.

## 4. Medium — correctness and timing use different external-baseline implementations

**Issue.** Exact same-ID conformance is established against the local `native()` helper, while the matched-service timing baseline is a separate imported implementation whose outputs are not checked here. This leaves a bridge missing between the correctness baseline and the timed baseline.

**Evidence.** `gpu_study.py:65` constructs `ExternalEvidenceScorer`. Lines 153-156 obtain same-ID output/state comparisons through `native(layout.public_ids, trace=True)`, not through that service. Lines 176-184 time `external.score(wire)` and discard its return value. I did not inspect the imported comparator, so this is not a claim that its computation or protections are wrong.

**Correction.** Add an explicit, frozen service-conformance check that the timed external service produces the same margin as the internal service and direct same-ID reference on all accepted requests, and document its enforced authentication/layout/lock/model-guard contract. Keep service overhead and direct-kernel timings distinct. If stage tracing is available for the service, verify those stages as well.

## 5. Medium — ordinary deletion's utility is omitted despite available predictions

**Issue.** The practical character-deletion comparator is evaluated for token/score inequality but not credential accuracy. A score difference alone cannot establish which encoding better preserves useful evidence or makes better task decisions. This leaves the practical baseline comparison incomplete even though the manuscript correctly avoids claiming that deletion is unfair.

**Evidence.** `gpu_study.py:158-162` stores `external_text_margin` for every accepted case. `analyze.py:49-54` reports only equality counts for deletion, while reporting accuracy and family-paired changes only for native versus compiled scoring. All 192 cases are accepted in the current cohort, so the existing records should support a directly comparable deletion-accuracy analysis without further GPU inference.

**Correction.** Add native, compiled/same-ID, and character-deletion accuracy together, with paired differences by the eight task families, and report prediction disagreements and prompt/encoding differences. Preserve the distinction between fidelity to native outputs and correctness against the explicit credential labels. Label any newly chosen endpoint as post hoc.

## 6. Medium — exact-stage comparison can silently ignore missing stages

**Issue.** The direct comparison code uses `zip` and `all` without verifying stage count or stage identifiers. A shortened or empty trace can therefore yield a successful comparison of an incomplete prefix; `all([])` is true. This is a robustness gap in a central exact-conformance endpoint, not evidence that current traces were actually truncated.

**Evidence.** `gpu_study.py:150-157` compares zipped state lists and increments comparison counts by the resulting list length. The manuscript section 4.2 states that all 31 retained scoring stages are compared. `analyze.py:30` checks equality of fingerprint tuples between names, but does not assert the expected number of stages; consistently short tuples could pass. The current aggregate counts are consistent with the reported 31-stage protocol, but aggregates alone are weaker than per-case structural assertions.

**Correction.** Require equal lengths and the model-derived expected stage count before comparisons; store ordered stage names, shapes, dtypes, and finite-value checks with each fingerprint. Assert exact per-stage shape equality and count all planned stages in analysis. Check that every planned context/name pair appears exactly once, not merely that there are 192 total rows (`analyze.py:16`).

## 7. Medium — finite-panel construction limits the statistical interpretation of admission gains

**Issue.** Disjoint name strings prevent literal fitting/test overlap but do not create a representative name sample or independent layout observations. The tokenizer experiment crosses 24 constructed names with eight layouts and reuses a fixed evidence sentence. Its aggregate 6.25-point gain is an exact descriptive result on that panel, not an estimated population admission effect.

**Evidence.** `boundary_study.py:23-27` crosses layouts/names using the same appended `Evidence: active license.` text. `analyze.py:60-61` sums layout counts into 192 held-out variants. The manuscript correctly calls the tokenizer study exploratory and the names constructed. The GPU study's different cohort has 192/192 admission under both methods (`summary.json`), so it supplies no replication of a coverage improvement.

**Correction.** Keep the finite-panel claim explicit wherever the percentage is reported. Provide the full paired layout/name table and report gains separately by layout. For generalization, predeclare independently sourced names, representative surrounding CV text, role placements, multiple name fields, and a sampling scheme; analyze repeated names/layouts as repeated measures. Do not attach a binomial interval based on 192 independent applicants or infer racial admission parity from this panel.

## 8. Medium — freeze provenance is narrower than a frozen complete analysis

**Issue.** The local plan is useful, but it does not freeze every artifact controlling the analysis or fully establish temporal priority to all prior model exposure. A hard-coded date and `before_inference: true` are local assertions. The manuscript's explicit distinction from externally registered preregistration is therefore essential.

**Evidence.** `gpu_study.py:43-58` hashes `src/*.py`, the study source, and selected dependencies, and writes the plan before the inference loop. It does not include `analyze.py` or `audit/checkpoint-manifest.json` in those recorded source/dependency dictionaries. The current analysis hash appears only in the post-inference summary. The plan lists broad endpoints but not detailed bootstrap aggregation, missing-data handling, or all analysis rules. The code fixes `arc_path` to a local-private file (`gpu_study.py:39`); the plan records its hash and IDs, which identifies a snapshot but is not a retrieval recipe by itself.

**Correction.** Preserve the existing plan as a prospective local inference freeze, and publish a separate manifest binding the analysis, checkpoint/tokenizer manifest, dependency versions, commands, and source/data retrieval recipe. Distinguish predeclared endpoints from later analysis and follow-ups. Use an externally timestamped immutable record for any future preregistration claim. The manuscript says reconstruction and ARC provenance are published elsewhere; those claims require checking those artifacts, which this review was not authorized to inspect.

## 9. Medium — capability control and identity-review validation have restricted scope

**Issue.** The ARC control exercises an empty-identity/all-evidence route, while the annotation protocol is still an unconducted human-review plan. These cannot validate the utility or annotation reliability of removing applicant names in real CVs. This limitation is mostly stated accurately; avoid any broader preservation or hiring-readiness phrasing elsewhere.

**Evidence.** `gpu_study.py:198` submits ARC prompts with no identity spans. `summary.json` reports 100/100 full-vocabulary matches and 68/100 accuracy for each path, whereas credential accuracy declines from 179/192 to 168/192. The protocol begins `Status: prepared, not conducted` and says authentication signatures are not evidence of review. The manuscript acknowledges these distinctions and describes the pending AI review as automated.

**Correction.** Present ARC as an all-evidence-path regression check. Validate the actual intervention separately on realistic CV/job tasks with qualified reviewers, an explicit definition of identity versus job evidence, uncertainty handling, frozen original annotations, disagreement reporting, and domain labels. Report any AI annotation exercise as automated synthetic consistency review only. The present human-review plan must not be counted as completed human evidence.

## Overall assessment

The supplied evidence supports a narrow deterministic conformance and finite tokenizer-feasibility report. The current manuscript is substantially more careful than a claim of racial-bias erasure or hiring-utility preservation, and I found no basis in these materials to assert fabricated results. The clearest remaining scientific corrections are the fixed-anchor matching ablation, the timing unit/estimand corrections, and practical deletion-baseline utility. The source and analysis checks above should strengthen exactness claims before external review. None substitutes for independent human role/task validation, independent execution, or broader empirical replication.
