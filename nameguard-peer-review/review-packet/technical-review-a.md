# Automated technical review A

Reviewer: AI agent, 9 November 2026. This is a separate automated technical review after sealing the blinded annotation. It is not independent human peer review. The sealed annotation was not edited, and reviewer B was not inspected.

## Scope and assessment

Only these six sources were read for this phase: `paper/THEORY.md`, `paper/manuscript.md`, `src/compiler.py`, `tests/test_compiler.py`, `fidelity_audit.py`, and `results/summary.json`. No runtime dependencies, per-case result rows, other tests, or cited publications were inspected. Consequently, this review cannot authenticate the GPU arrays, semantic approval process, literature priority, or fidelity aggregate counts.

The principal finite-set optimization and finite-real-score fidelity formulas are correct under their stated mathematical model. The manuscript already acknowledges their elementary nature and does not generally overclaim mathematical novelty. The most consequential issue is the interface between structural feasibility and actual reviewed admission. That distinction needs to be made exact before a support certificate is described as an admission guarantee.

## Findings

### A1 — High: structural support is equated with admission while witness approval is still pending

**Evidence.** THEORY lines 15–21 permit only explicitly reviewed formatting reclassification, assert that membership in A_i is equivalent to admission, and then say every selected witness still requires semantic approval. In contrast, `enumerate_candidates` (compiler lines 26–51) deliberately grants no semantic approval and enumerates whitespace expansions using only a native-layout preflight. `select_contract` (lines 54–72) counts every such candidate without approval information. `match_contract` (lines 75–79) returns the cheapest unresolved match. Manuscript lines 95–107 call this actual admission and an exact rejection floor; line 167 calls the enumerated candidates approved.

**Issue.** The optimization theorem is correct for the candidate sets provided, but the implementation optimizes structural support. If some formatting reclassification is disallowed by semantic review, structural support need not equal realizable service coverage. For example, a contract structurally supported by three members can become unsupported after review while a different contract supported and approved for two remains feasible. Moreover, selecting the cheapest unapproved witness can reject even when a more expensive approved witness with the same evidence exists. This is not evidence that the recorded cohort was wrongly admitted; the permitted sources do not establish whether all its counted witnesses were actually approved.

**Correction.** Define separate structural sets C_struct and approved sets C_approved. Optimize over approved witnesses for an actual-admission optimality claim, or explicitly report structural maximum support as an upper bound conditional on later approvals. Actual admission must require both an exact contract match and approved witness provenance. If review happens after selection, enumerate matching witnesses for review rather than treating the cheapest unresolved one as definitive. Report which notion each count measures and bind approval to source, interval, tokenizer, and policy.

### A2 — Medium: the fidelity audit does not establish complete planned-family coverage

**Evidence.** `fidelity_audit.py` lines 23–32 group only rows present in the input. Its full-family check compares accepted rows with that observed block. There is no check for expected context IDs, six distinct naming variants per context, duplicate case keys, or the complete 192-row plan. The supplied summary declares 192 planned rows and 32 contexts.

**Issue.** Dropping one or more rows, deleting an entire context, or duplicating a name can still pass the full-family-admission check and produce a seemingly valid fidelity audit. Duplicate rows also alter the MSE and minority-count weights. The audit currently establishes admission only for observed rows, not the planned family. It cannot independently substantiate the manuscript's complete-panel counts from these checks alone.

**Correction.** Bind the audit to a declared case manifest and verify exact case-key coverage, uniqueness, context membership, and name membership before computing results. As a minimal cohort-specific check, require exactly 32 contexts, six distinct planned names in every context, and 192 unique planned cases. Include the source-row digest and cohort verification result in the audit artifact. Preserve the current explicit refusal of partially admitted observed families.

### A3 — Medium: the compiler's whitespace fixture contains an incorrect applicant span and misses the intended left-expansion exercise

**Evidence.** `tests/test_compiler.py` lines 44–51 use `text = 'Name:  Ada \nEvidence'` with `enumerate_candidates(text,8,11,...)`. In exact Python code-point offsets, `text[8:11]` is `'da '`. The complete name is `[7,10)`. Character 7 is substantive `A`, so the submitted start also blocks all left whitespace expansion.

**Issue.** The test can pass while its reviewed identity interval excludes part of the name. It is not a meaningful check of complete-name removal and does not exercise left expansion in this fixture. Its existence/non-substantive-expansion assertions do not test enumeration completeness or budget edges.

**Correction.** Set the span to `[7,10)`, update the slice assertions accordingly, assert the selected name substring, and assert the exact expected candidate boundary set under the character tokenizer. Add a focused finite tokenizer fixture with cross-boundary token merges to check that native factorization filters candidates. A character tokenizer alone cannot exercise BPE boundary infeasibility. These are targeted correctness checks, not a demand for broad testing without a failure model.

### A4 — Medium clarification: “complete observable outcome” is complete only for a deliberately restricted public interface

**Evidence.** THEORY lines 25–31 define the observable as a rejection tag or an acceptance tag plus scalar score. Manuscript lines 128–129 and 304–308 explicitly allow raw embeddings, length, and timing to differ and place these within the trusted host.

**Issue.** Proposition 1 is correct for this defined O, including its total-variation lower bound. It is not a theorem about all observations an actual applicant or API client could make. Distinct rejection messages, response timing, returned token counts, or metadata would enlarge O. The current limitations substantially acknowledge this, but the word complete can still suggest a stronger interface guarantee than the definition provides.

**Correction.** Call it the complete *decision-and-score* outcome and specify the public observable interface and threat model alongside the definition. Either explicitly exclude metadata, timing, and rejection reasons from the claim or incorporate their public values into O. Audit uniform rejection behavior if the deployed interface exposes different failure paths. No claim of a concrete runtime side-channel defect is made here, because that implementation was outside the permitted review sources.

### A5 — Low: the fidelity implementation needs finite-value validation and scale-aware numerical checking

**Evidence.** `audit` (lines 9–19) checks only that the sequence is nonempty. `main` uses Python JSON output without `allow_nan=False` and checks the decomposition with a fixed absolute tolerance of 1e-12 (lines 33–41).

**Issue.** NaN or infinity can enter arithmetic and comparisons, make sign counts meaningless, and lead to non-standard JSON numeric values. A fixed absolute tolerance can fail on large finite scores due to cancellation despite a correct mathematical identity. These are robustness gaps, not demonstrated failures on the reported bounded-margin cohort.

**Correction.** Require finite real scores and finite constants, reject booleans as score values if strict typing is intended, serialize with `allow_nan=False`, and use a documented absolute-plus-relative comparison. Define population variance explicitly as n^(-1) sum_i (v_i-mean(v))^2 and define the native positive label as v >= 0, including ties, to match the code.

### A6 — Low: public optimization helpers accept values outside the theorem's candidate domain

**Evidence.** `Candidate` has annotations but no runtime validation. `select_contract` accepts a candidate with formatting_cost=-1 and returns total_formatting_cost=-1. The theory requires nonnegative cost, and compiler-generated candidates satisfy it.

**Issue.** This does not break the selector for trusted output from `enumerate_candidates`, but the public helper's guarantee depends on an unstated valid-input precondition. It also accepts fabricated token tuples and spans without source binding.

**Correction.** Either declare the helpers internal and state the trusted-candidate precondition, or validate candidate types, nonnegative costs, token-ID domain, and provenance before an externally supplied candidate set is used as a certificate. Do not imply that dataclass type annotations enforce these constraints.

### A7 — Low mathematical wording: state the statistical and empty-set edge assumptions explicitly

**Evidence.** THEORY lines 125–127 call the zero-failure formula exact for independent Bernoulli trials but derive it using one common p. Lines 75–77 use max_e s(e) without restating the E=empty convention.

**Correction.** Say independent trials with a common Bernoulli failure probability p (or i.i.d.) for the exact binomial bound. Independence alone allows unequal p_i, where the no-failure probability is product_i(1-p_i), not (1-p)^n. If bounding their arithmetic mean by this formula under independence, explain the AM–GM argument and its conservative rather than generally exact character. Define the empty-union support maximum as zero in the coverage corollary, or explicitly treat that case separately as Proposition 2 already does.

## Checked mathematics and implementation

I extracted only the allowed pure definitions by AST, without importing project dependencies, and performed the following separate checks:

- Exhaustive comparison of all three selector objectives over 625 four-member panels formed from five small candidate-set options. All 624 nonempty-union panels matched an independent support/cost/token-order objective; the empty-union panel rejected. The test included duplicate witnesses and different per-member minimum costs.
- Direct checks of the MSE decomposition, minimax-midpoint identity, and minority decision-change bound on four finite score families, including a threshold tie and a singleton. All passed.
- Direct offset inspection confirmed the test-fixture error in A3. A negative-cost candidate confirmed the interface precondition gap in A6.

For exact fixed sets, deduplicating each member before accumulating support is correct. The cost tie-break is the sum of memberwise minimum costs, not a sum over duplicate witnesses. The (b+1)^2 enumeration bound is correct for one interval with at most b characters of expansion on each side. It is a bound on boundary pairs, not a tokenizer-independent runtime complexity guarantee; the current text already acknowledges tokenizer cost.

Proposition 3 is a correct conditional induction when the displayed recurrence is the actual computation: equal retained IDs, positions, masks, frozen parameters, deterministic blocks, and no hidden context-dependent state imply equal aligned states and outputs. It is not established merely by G_x S_x=I; implementations must actually run the recurrence on gathered evidence, with fixed cache state, readout, and evaluation behavior. The manuscript's matched external baseline is mathematically decisive under those same assumptions, and appropriately limits claims for internal placement. I did not inspect the runtime to verify those assumptions.

The native-fidelity bounds are correct for finite real scores under the empirical uniform weighting. The binary lower bound assumes full-service deterministic invariant binary decisions with a fixed threshold/tie convention. It does not cover abstaining methods without a specified rejection loss. The mean of contextwise half-ranges is an average of per-context minimax floors, not a single global worst-case floor. The reported 0.44242 is consistent with half the supplied 0.8848302960 mean range, but the five/fifteen decision-change totals and individual MSE records cannot be independently recomputed from `summary.json` alone.

The manuscript correctly separates score fidelity from accuracy. With one correct qualification label per context, an invariant binary rule can simply output that label. Hence the observed 11/192 net accuracy loss is not forced by the fidelity lower bound. With the stated binary full-service setup, fifteen changed decisions and an eleven-error increase are arithmetically consistent with thirteen formerly correct predictions becoming wrong and two formerly wrong predictions becoming correct; this is an algebraic consistency observation, not a row-level verification.

## Novelty and evidence boundary

The theorem statements do not constitute new general optimization, information-flow, or projection theory, and the current draft largely says so. A defensible contribution is the specific systems formulation, implementation, source-bound policy, and audit. The strong equivalent external baseline also means exact internal placement cannot itself be offered as the source of superior fairness on matched evidence. No field-wide priority can be assessed from the allowed local files; the bibliography was not externally checked in this restricted review. Exact conformance on a constructed panel does not replace human role annotation, real qualification validation, multi-runtime evidence, or independent replication.

**Recommendation:** resolve A1 before presenting structural support as actual reviewed-admission optimality; fix A2 and A3 as concrete audit/test defects; preserve the existing conservative novelty and accuracy language. The other findings tighten assumptions and robustness rather than refuting the elementary core results.

## Author-reported remediation after this review

The coordinating project author reports correcting the Ada test interval to `[7,10)`, adding strict cohort completeness checks, and preparing a theory/manuscript clarification that the compiler certifies structural token-witness support while actual admission additionally depends on semantic approval. These are author-reported responses, not independently revalidated changes. This review preserves its findings against the version read above; no revised source or reviewer B material was inspected.
