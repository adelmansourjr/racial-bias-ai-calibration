# Recovering verified credential coverage without restoring name influence

The previous study verified **32 of 48** cases with a definite source-backed
answer. It referred the other sixteen because the model's two highest-ranked
records did not include a controlling record. A verifier can prevent a wrong
definitive decision while still missing a correct one. This follow-up changes
proposal construction and measures both effects separately.

## Policy change

A record can control the decision only if it comes from the designated
registry, belongs to the applicant, matches the exact credential, and is
effective by the cutoff. Within those records, use the latest effective date.
Use the frozen, protected Qwen logits only to break ties between records at
that date. Propose one record. The unchanged full-ledger verifier checks the
proposal and emits the decision, reason and controlling record IDs. It refers
missing or conflicting evidence and reports the conflicting record IDs.

The policy uses the same input and five answer scores as the earlier model
ranking. The intact named prompt reaches tokenization and embedding. The
inherited gate zeros the identity positions and runs the native blocks on
ordered evidence rows, with fixed evidence-relative coordinates. No score
calibration, weight update, name rewriting or output averaging is needed.

One-proposal coverage follows from the rule, given a complete and correct
supplied authoritative ledger. Let (A) be its applicable records and (L)
the records in (A) with maximum effective date. A definitive case has a
nonempty unanimous (L), so any record selected from (L) is an admissible
witness. A priority ordering by date always selects such a record. When
(A) is empty or (L) conflicts, the verifier refers independently of model
scores. An arbitrary wrong model ranking cannot create a wrong definitive
report. It also cannot prevent a verified decision if a definitive record
exists under this contract.

Name invariance uses a distinct argument. Under approved role boundaries and
equal ordered retained tokens, the deterministic protected computation
produces the same retained embedding rows, positions and masks. Induction
through the native blocks gives equal aligned activations and logits.
Given the same ledger and exact same logits, the fixed filtering, ordering,
tie-break and serialization yield identical report bytes. Admission can
depend on layout when tokenizer boundaries fail, so admission is part of the
test denominator. Raw pre-gate name embeddings still contain name information.

## Experimental controls

The development run used three credential families under two names. The
held-out benchmark uses twelve other families, eight conditions per family,
and four names. Each condition contains four registry records. It receives
fresh opaque IDs and a shuffled display order. The sole controlling record
occupies each display position exactly twelve times among the 48 decisive
cases. Definite and unresolved cases have the same number of records.
Role swaps, status changes, later invalidation, wrong credentials, missing
applicant evidence, future-only evidence and same-date conflicts test factual
sensitivity rather than name equality alone. Synthetic fixture labels are
derived from the edit and do not enter the prompt or ranking algorithm.

We reuse each score vector for the original top-two rule, a top-two rule after
applicability filtering, and the one-proposal date-first rule. This is a
within-case ablation with a fixed prompt, model, evidence input and checker.
An independent direct registry evaluation is the deterministic comparator.
Sorting opaque IDs and taking the first two displayed records are negative
controls. Expected uniform top-two recall is one half because each decisive
case has exactly one controlling record among four. The earlier ID-based
shortcut therefore has explicit controls in the fresh design. Position
sensitivity in language-model selection has prior evidence.
[Zheng et al.](https://arxiv.org/abs/2309.03882)

The plan freezes source and metric hashes before inference on each set. The
primary utility count uses one predeclared name per condition. Name swaps are
exact equality observations. A paired bootstrap resamples credential
families for a descriptive interval, with related conditions kept together.
The activation audit directly compares aligned arrays by bit pattern on the
planned subset and checks zeroes in designated identity rows. The test counts
each missing prediction against coverage.

## Archived replay

On the previously published logits, the original method reproduced **32/48**
definitive cases. Both ranking among applicable records and proposing the
latest applicable record recovered **16** missed cases, reached **48/48**,
and retained all 48 correct review decisions. Across the original four
names, all 96 conditions kept identical outputs and score vectors. This is an
archived replay used to verify that the proposed change addresses the actual
published failure.

## Held-out results

| Method | Correct definitive (48) | Correct reviews (48) | Wrong definitive | Maximum proposed records |
|---|---:|---:|---:|---:|
| Original model ranking | 22 | 48 | 0 | 2 |
| Filter for applicability, then model ranking | 29 | 48 | 0 | 2 |
| Latest applicable authoritative record | **48** | **48** | **0** | **1** |
| Complete ledger rule, without the model | 48 | 48 | 0 | 0 |

The repaired policy recovered 26 missed decisions, with zero losses on a
matched comparison. This is a 54.17 percentage point gain among decisive
conditions. A descriptive paired bootstrap over twelve credential families
gives an interval of 45.83 to 62.50 percentage points. The original scorer got
6/12 confirmations and 16/36 contradictions; the repaired policy got 12/12
and 36/36. All 96 repaired statuses and controlling ID sets match the fixture
truth, including all same-date conflicts. Both sides of each of the seven
fact contrasts were correct in all twelve families. The display-order
baseline covered 24/48 decisive cases and sorted opaque IDs covered 20/48.
Uniform two-record ranking has an expected coverage of 24/48. Removing the ID
shortcut is therefore material to interpreting the original model ranking.

All **384/384** planned case-name inputs were admitted. All **96/96**
conditions have bit-identical five-letter logit vectors and byte-identical
reports, independently across each of the original, filtered and repaired
policies. On the frozen trace subset of eight conditions, the three
counterfactual comparisons per condition match on each of 31 arrays, for
**744/744** direct comparisons. All **960** designated identity-row zero
checks pass. The frozen model files and parameter fingerprint verified
after inference.

Every predeclared gate passed. Applicability filtering alone leaves old
but genuine records in competition for two slots. The latest-date rule
removes that failure and preserves valid review on conflict or absence.
The unit tests additionally enumerate all 120 strict orderings of the
five possible answer logits on all 96 fixture conditions, exercise
same-date unanimous ties and test untrusted sources. Those CPU checks test
the policy; the frozen GPU observations separately test actual name
invariance.

## Scope of the guarantee

The coverage gain comes from using verified record metadata in proposal
construction. The model's ability to infer dates, identity roles or issuer
authority from unreviewed prose has not improved by this experiment.
The deterministic comparator achieves the same results without a model.
The verifier and proposal layer have access to the full supplied ledger.
One proposed witness does not imply fewer ledger records were inspected,
lower latency, or fewer external record accesses.

The model-assisted prompt supports at most four selectable records. The
tests use simulated requirements with clearly specified authority and an
equal-precedence policy. A record missing from a real ledger, a mistaken
date, a disputed authority, or an incorrectly annotated name can invalidate
the premises. The registry lookup and holder linkage must also be fixed across
name interventions; an upstream lookup that changes the ledger is outside this
comparison. A real evaluation would need independently reviewed role
spans, authorized credential records and a separately justified decision
policy. The test measures a specific name intervention and carries no
population estimate of demographic disparities.
