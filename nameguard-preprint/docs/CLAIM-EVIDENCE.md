# Claim and evidence ledger

This revision adds literature and claim checks to an existing frozen experiment.
It does not create additional model measurements or change base weights.

| Claim | Evidence and boundary |
|---|---|
| Equal Q implies equal deterministic score | Composition and induction in paper Section 5; exact roles and an isolated scoring branch are assumptions. Z3 checks an abstract congruence specification, not the Python implementation. |
| Public scoring activations match | 108 inherited forwards across only three generated bodies; 105 counterfactuals x 32 stages = 3,360 comparisons. Raw source activations are excluded. |
| Native evidence computation is preserved | Six inherited hidden/vocabulary comparisons on three bodies; architectural equality uses the identical T(Q), positions and weights. Does not imply equal named-CV behavior or useful grading. |
| Evidence changes can affect scores | Nonconstant outputs across three bodies; not a validation of hiring accuracy. |
| Scoring token count is native | By construction |T(Q)|; 32 inherited generated cases confirm bookkeeping. Total work includes |T(X)| extra embeddings. |
| Service speedup is 4.10x locally | 21 inherited paired warm trials against scalar service; not against optimized external anonymization or an isolated tokenization ablation. |
| Authentication catches tampering/declared omissions | 108 registered-omission and 36 conflict controls; no human semantic accuracy measured. Unregistered alias can survive. |
| Behavioral equality does not imply information erasure | Exact finite witness in scripts/scientific_claims.py; compatible with probe/erasure literature [13-16,25]. |
| Internal placement improves fairness over equivalent external Q | **Not established; equal Q and identical scoring computation imply equal outputs.** Raw embedding first adds no mathematical score-isolation benefit. |
| Whole-model names/race information is removed | **False for this architecture:** original weights and raw embeddings remain. |
| No racial bias remains | **Not established:** other proxies, job policy, review errors and deployment outcomes remain. |
| Universality for every raw CV without metadata | **Not established:** semantic role correctness is an external assumption even when enforcement is internal. |
| Ready for publication | A reviewable preprint draft/artifact; not peer reviewed, independently replicated, or certified novel. |

## Necessary next empirical study (not completed)

1. Obtain ethically sourced/licensed real documents with independent source-role
   annotations and adjudication. Split by evidence document and name family,
   preventing variants of one CV from crossing train/development/test boundaries.
2. Compare raw scorer, equivalent external Q scorer, scalar internal scorer and
   canonical internal scorer. Include learned adapters/erasers only if trained
   with an independent development split; freeze selection before the final test.
3. Report document-level omitted-identity and false-evidence-removal rates;
   applicant corrections and ambiguous/unknown-alias cases must be retained.
4. Evaluate held-out job-relevant labels and qualification-sensitive pairs.
   Prespecify an acceptable utility noninferiority margin with domain experts;
   test paired accuracy differences at the independent document/job level.
5. Name swaps are repeated measurements. Use CV/job-clustered intervals or
   permutation tests, report worst-case per-document ranges, and reserve a fresh
   confirmation set after any tuning. No significance claim follows from treating
   3,360 correlated activation comparisons as 3,360 independent applicants.
6. Audit identity recovery separately from output effects. Use held-out linear
   and nonlinear probes with controls; distinguish public from private states.
   Probe failure is bounded evidence about that decoder and distribution.
7. Benchmark actual end-to-end external-Q and internal-Q services with matched
   integrity checks, explicit preprocessing/signing costs, workloads and hardware.

Do not preregister a completed experiment retrospectively. The inherited local
source freeze is not an independent public preregistration.
