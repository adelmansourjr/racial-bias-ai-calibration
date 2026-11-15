# Recovering credential accuracy after name isolation

*Research addendum, 16 November 2026*

This study investigates a concrete failure in the earlier [admission-aware preprint](../nameguard-peer-review/paper/manuscript.md). Its frozen Qwen scored 179/192 native and 168/192 with the name-isolated readout on controlled credential cases. The weaker performance was concentrated in double negation. A new prompt recovered the earlier loss in a post-hoc replay. A separately fitted readout also improved a harder, held-out comparison while preserving exact name invariance. Missing evidence remained the most consequential failure.

## The question

Can we keep the result identical when only a designated applicant name changes, while responding correctly to changes in what the CV actually establishes? We test three different properties: whether the scoring states carry information about the designated name, whether the observed score and report change under a name substitution, and whether the score corresponds to the declared credential rule. The first two permit an exact structural proof. The third requires independent task evidence.

A model may return the same wrong answer to everyone. Our controls change who has the credential, when it was valid, or whether the evidence suffices. Several matched reversals preserve every case-folded word and its count while changing the correct answer. A constant answer or a bag-of-words rule cannot solve both members of such a pair.

## Model intervention and report

The original named prompt reaches the frozen Qwen tokenizer and embedding lookup intact. A reviewed role packet designates the applicant name. The gate zeros those activation rows within the scoring execution and runs native blocks on the exact retained evidence tuple. The approved tuple is fixed before the incoming name is scored. This study uses one affinely quantized, 1.5B Qwen derivative on Apple MLX. The file and parameter digests were verified after each GPU run.

For a fixed prompt and evidence contract, the induction in [the formal analysis](MATHEMATICS.md) gives equal aligned scoring states for admitted name variants. Fixed aggregation, a frozen readout, and deterministic serialization preserve that equality. The observed report is a short explanation of the model's assigned status. It consumes the credential and status only. It makes no claim to expose the model's reasoning. A second, structured-record interface can cite exactly which verified records controlled a policy outcome.

The new classifiers distinguish three outcomes:

- **Confirmed** when the model judges the required credential current.
- **Contradicted** when the model judges that the evidence establishes inactive or absent status.
- **Review** when available evidence does not support one current status.

A review referral is counted as incorrect on a known-status case. On missing or conflicting cases it is the expected outcome. This prevents an all-referral system from being reported as an accurate credential scorer.

## Experiment 1. Diagnose and replay the historical cohort

The historical diagnosis uses the original 192 logged rows. Each of 32 CV conditions was scored under six names. The unit of inference is the CV condition or task family. The per-name row counts below describe the experiment's outputs.

| Wording | Target | Native correct | Old protected correct |
|---|---|---:|---:|
| Direct | Inactive credential | 41/48 | 42/48 |
| Direct | Current credential | 48/48 | 48/48 |
| Negation | Inactive or absent credential | 48/48 | 48/48 |
| Negation | Current credential | 42/48 | 30/48 |
| All | | **179/192** | **168/192** |

The positive double-negation group accounts for the deficit. This localizes a wording-sensitive failure. It does not establish a unique mechanism for the native versus isolated difference.

We replayed all 192 rows with the same CV bodies, job credentials, and names. The revised prompt gives six short examples across affirmative, negative, missing, and conflicting evidence. It changes the surrounding instructions and adds an explicit review class. In this **post-hoc** replay, accuracy increased to **180/192**. Twelve cases went to review and were counted incorrect. All definitive decisions were correct on this particular panel. The rewritten prompt got **48/48** positive double-negation rows right. All **32/32** groups had identical three-label logits and serialized explanations across the six names.

Because this cohort informed the diagnosis, the replay demonstrates repair of that measured failure. It serves a different purpose from the fresh evaluations below.

## Experiment 2. Test balanced labels and a review option

Six development credential families supplied 36 scenarios. The frozen test uses 24 disjoint families and 144 new scenarios. Each family contributes two positives, two negatives, and two unresolved cases, with one primary name per scenario for utility. Test phrases differ from development phrases. A predetermined subset runs all four test names. Each name swap is a conformance run and does not increase the utility sample size.

The comparison runs native named scoring, the previous protected prompt, a clarified binary prompt under each answer mapping, its paired margin, and an external name-deletion control using the same binary pair. On all scenarios it also measures a single three-way prompt and the same prompt cyclically rotated across letters A, B and C. The binary rule uses the original zero threshold. The paired method subtracts the reversed margin from the forward margin and halves the difference. Three-way rotation averages logits after putting them back into the same semantic order.

The label-rotation tests distinguish additive label bias from other prompt effects. Rotation cancels a common additive offset under the decomposition in the formal analysis. The remaining interaction can still reduce actual accuracy. Label swapping, prior correction, and contextual calibration have direct precedents. [Zheng et al.](https://arxiv.org/abs/2309.03882) · [Zhao et al.](https://proceedings.mlr.press/v139/zhao21c.html)

| Method | Known statuses correct (96) | Correct referrals (48) | All correct (144) | Incorrect confirmations on unresolved cases (48) |
|---|---:|---:|---:|---:|
| Native binary | 73 | 0 | 73 | 30 |
| Previous protected binary | 71 | 0 | 71 | 21 |
| Clarified binary, forward | 72 | 0 | 72 | 1 |
| Clarified binary, reverse | 51 | 0 | 51 | 0 |
| Protected paired binary | 61 | 0 | 61 | 0 |
| Character deletion plus paired binary | 57 | 0 | 57 | 0 |
| Single protected three-way prompt | 73 | 0 | 73 | 48 |
| Cyclic protected three-way prompt | **83** | **8** | **91** | 40 |

The primary paired binary intervention **failed** its intended utility rescue.
It lost 12/96 known-status decisions relative to native scoring. The descriptive
family-bootstrap interval for the change was [-19.79, -6.25] percentage points.
The single forward mapping worked better than its paired version, 72/96 versus
61/96. Both paired approaches missed every positive-negation case. The 61/96
versus 57/96 result for internal versus character deletion applies to that prompt
and token encoding only. With identical retained token IDs, the earlier
[matched external comparison](../nameguard-peer-review/paper/manuscript.md)
gave exactly the same states and scores.

Cyclic rotation improved its matched three-way prompt by 18/144 overall. The
descriptive family-bootstrap interval was [+6.94, +18.06] percentage points.
Its strongest result was 83/96 on evidence that established a definite status.
It correctly referred only eight unresolved cases and falsely confirmed forty.
In particular, it falsely confirmed every conflicting record. These are unsafe
error patterns for credential verification, even with name-invariant scores.

All 252 planned named requests were admitted under the six prompt contracts.
In all 36 conditions checked across four names, paired scores, cyclic scores,
serialized explanations and every aligned scoring-stage fingerprint matched.
We recorded 20,088 stage fingerprint comparisons and 25,920 checks of zeroed
identity rows. Fingerprints are equality diagnostics for the measured finite
arrays. The recurrence supplies the mathematical equality result under the
stated runtime assumptions.

## Experiment 3. Test who holds the credential and when

The compositional study is frozen separately. It fits a three-class ridge readout to 48 isolated final activation vectors from six fitting credential families. The transformer supplies the representation. The fixed regularization parameter is 0.1. The fit occurs before held-out inference, uses no actual applicant records, and produces no merged adapter. Twelve new credential families contribute 96 test cases, with different wording from the fitting cases. The shared task constructs are deliberately similar to those in fitting, while subjects, credential strings, wording and surrounding facts change.

Each test family contains a subject reversal, a date reversal, a negation reversal, missing evidence and conflicting evidence. Subject and date reversals preserve the case-folded word multiset. Pair success requires two correct decisions. This avoids interpreting a favorable aggregate score as evidence of comprehension when the classifier returns the same answer on both members.

| Method | Known statuses correct (72) | Correct referrals (24) | All correct (96) | Incorrect confirmations on unresolved cases (24) |
|---|---:|---:|---:|---:|
| Native binary | 41 | 0 | 41 | 19 |
| Previous protected binary | 42 | 0 | 42 | 20 |
| Plain protected three-way prompt | 36 | 0 | 36 | 0 |
| Few-shot protected three-way prompt | 44 | 8 | 52 | 14 |
| Ridge readout on protected activations | **57** | **12** | **69** | 12 |

The learned readout improves total accuracy by **27/96** over the previous gate with its original readout, or **28.13 percentage points**. A descriptive paired bootstrap over the twelve constructed credential families gives **+20.83 to +35.42 points**. On known cases it improves from 42 to 57 correct. On matched reversals it gets both sides correct on 8/12 subject pairs, 8/12 date pairs and 5/12 negation pairs. The native comparison gets 5/12, 0/12 and 0/12 respectively. The few-shot method is stronger on subject pairs at 9/12 and has no fully correct date pair.

The best readout correctly refers all twelve conflicting cases. It falsely confirms all twelve missing-evidence cases. The few-shot method correctly refers eight of the missing cases and incorrectly confirms eleven of the twelve conflicts. This limits the use of either readout for credential verification from unreviewed text. Perfect use of the declared rule ought to send all twenty-four cases to review. A model-generated positive result requires independent corroboration when evidence is uncertain.

Across **24/24** tested conditions with three names, the final protected-state fingerprints, ridge scores, both protected prompt logit vectors and deterministic ridge explanations match exactly. The ordinary replay separately tests six names per condition. The row-level records distinguish fingerprints from direct floating-point comparisons of scores and logits.

This is a held-out result for different task families and expressions, using constructed examples authored for the study. The synthetic benchmark and dedicated data splits support controlled mechanism testing. A public benchmark about broad natural-language inference would answer a separate question, and neither would supply real qualifying decisions without suitable labels. Matched reversal designs also have antecedents in stress tests for superficial inference heuristics. [McCoy, Pavlick and Linzen](https://arxiv.org/abs/1902.01007)

## Experiment 4. Check a rule on reviewed structured records

The separate evaluator takes the required credential ID, evaluation date, and reviewed findings keyed to holder role, credential ID, date, and record ID. It uses the latest applicable equal-authority records, ignores other people, unrelated credentials and future records, and sends missing or conflicting evidence to review. Its report cites sorted controlling record IDs and the reason for its decision.

An exhaustive finite check covers **1,024** combinations of old and current evidence, third-party records, unrelated credentials and future records. Every outcome and controlling record set agrees with the fixture oracle. It also passes **4,096** serialized report comparisons over input order and names associated with the intake form. The evaluator's interface takes reviewed observations. It is a policy correctness check and adds no successful cases to the free-text accuracy counts.

## What is substantial and what is original here

1. **The result shows that the historical utility loss was recoverable.** A frozen name-isolated model can exceed its prior named baseline on the exact CV bodies and still emit the same logits and explanations under each tested substitution. The loss in the previous experiment was a property of the scoring configuration tested there.
2. **A stronger held-out check makes the improvement meaningful.** Name invariance might coexist with an evidence-insensitive scorer. The binding reversals test changes that should affect decisions, and the frozen-state readout improves accuracy on both sides of many such pairs. The remaining errors are visible by case type.
3. **The evidence interface has an exact accuracy ceiling.** In the finite setting, group examples by the retained evidence. The best possible invariant accuracy is the sum of each group's majority count divided by all examples. This shows exactly when information removal makes perfect task accuracy impossible, independent of the optimizer. The bound is an application of the standard empirical Bayes classification calculation, and the prompt rotation and ridge equations are established techniques. The contribution is the application to this interface and the demonstrated separation of exact name invariance, factual utility and refusal behavior.

This extends the earlier admission and activation work. It does not assert scholarly priority for the underlying equations or model operations.

## Limitations and disposition

| Limitation | Evidence or action | What would settle the remaining question |
|---|---|---|
| Prior utility drop | Replayed all historical cases. Recovered 12 rows and positive double-negation accuracy. | Fresh task families have separate held-out results. |
| Answer-letter artifacts | Held wording fixed when reversing mappings. Measured forward, reverse, combined and three-way outcomes. | Additional prompt formats and independently chosen extraction models would test transfer. |
| Shortcut recognition | Held out credential families and wording. Added matched reversals and checked both members. | Untouched document sources and human-approved reference labels. |
| Overconfidence under missing or conflicting evidence | Added review as an explicit outcome and separate false-confirmation counts. Implemented the verified-record rule and coded its proof boundary. | A reliable independently validated extraction or authorized issuer record process before treating a fact as verified. |
| General LLM performance | Previous work's 100 official ARC-Challenge questions remain a separate clean-path check. This follow-up evaluates a task-side readout. | Broader benchmarks across calibrated task dispatch and hardware if making a wider preservation claim. |
| Unequal admission | Audit each required prompt contract and count incomplete admission. The previous broader tokenizer study still has unsupported variants. | Different natural CV layouts, representative names and independent role annotations. |
| Correct role annotation | Source hashes and authentication ensure what was reviewed is what enters the gate. The new cases use supplied roles as generator truth. | Human reviewers, ambiguity reporting, and explicit documentation of retained third-party text or proxies. |
| Identical narratives | The fixed reporting schema preserves exact decision explanation bytes. The structured evaluator cites controlling records. | Separate evaluation if unrestricted generated explanations are required. |
| Real hiring validity | The target is an explicit simulated credential rule. No outcome labels for actual hiring were introduced. | Independent job-domain review and genuine task labels intended to assess suitability. |
| Generalization across architectures | One pretrained checkpoint, frozen throughout, and fresh held-out language families. | Independent replication on another model family and its native tokenizer. |
| Reproducibility | Public synthetic fixtures, all records, checkpoint manifest, local pre-inference hashes and read-only verifier. | An independent lab reproducing the runs and auditing authorship and claimed novelty. |

Qualification rules can themselves be unfair or fail to reflect work capability. A guarantee about the name field does not constrain other race proxies in retained evidence. The researcher still needs to justify the qualification label and the governing evidence policy. The mathematical result is appropriately scoped to the designed scoring interface. The empirical result is a measured improvement on credential inference, with an archival baseline replay and held-out controls.
