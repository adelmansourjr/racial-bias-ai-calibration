# Invariant names, sensitive evidence

*Research follow-up, 16 November 2026*

## Problem

In the [previous utility study](../nameguard-utility-study/REPORT.md), the
strongest trained readout improved accuracy on held-out credential scenarios,
but incorrectly confirmed every one of twelve missing-evidence cases. It could
emit identical unsupported answers under every name. This follow-up asks
whether the model can instead make repeatable proposals that support exact
name-invariant scoring and verifiable reports.

The intended claim has three independent parts. The protected model should
respond identically to a name change. It should still respond to edits that
change the underlying credential facts. A final definitive status should have
a verifiable source in the supplied registry. Equality alone does not give the
second or third property. A source checker alone does not eliminate
name-dependent proposal ranking or selective review.

## Intervention

I wrote a new prompt that asks for the controlling record, using this rule:

> Select a record only if it belongs to the applicant, names the exact required
> credential, and is effective on or before the cutoff. An invalid record can
> be the correct source. Among matching records use the most recent date.

The complete prompt specifies how to handle conflict and absence and includes
unrelated examples. The test keeps candidates visible as ordinary text. A
registry ledger supplies the exact facts and record IDs to an independent
checker. The outcome can be confirmed, contradicted or referred for review.
The machine-readable explanation includes the decision reason and controlling
record IDs. It is generated from the verified policy result.

The original prompt with its applicant name goes through the native tokenizer
and embedding lookup. The inherited activation gate zeros reviewed identity
rows and runs the same frozen transformer blocks on the ordered evidence
activations. The [formal proof](METHOD.md) separates name noninterference in
the scoring path from factual correctness relative to the trusted source.

## Design and decision rules

The development set has three credential families and 24 conditions. The
original top-letter approach correctly verified four of twelve decisive
development cases. It often picked “no record” even with a valid source ranked
near the top. Restricting the ranking to actual record pointers and considering
the first two covered eleven of twelve in development. I froze this policy,
its minimum 75% definitive-coverage threshold and the other held-out
criteria before the test run. The first prompt and the first decision rule
remain evaluated. The date-stamped local hash manifest is separate from the
subsequent source-ID follow-up.

The main test has twelve new credential families and eight conditions per
family. The evidence starts with a valid record, then varies status, deletion,
the subject, the exact credential, date eligibility, same-date disagreement
and a newer contradictory record. Four names cross all 96 conditions.
We compare native and protected runs of the exact same prompt, unchecked and
checked interpretations of the same logits, a shorter protected prompt, all
record proposal budgets, and an exact registry-only policy. Unchecked reports
use the record's listed validity and illustrate how a real but irrelevant
citation can mislead. They are not an official model or issuer baseline.

Each name substitution changes the applicant-name slot only. In the selected
32 conditions, the test directly compares the bit patterns of all 31 aligned
arrays across variants, including the vocabulary vector. Binary equality of the
recorded explanation is checked across every planned condition. We use a
single predeclared name for utility counts and resample credential families
for descriptive paired uncertainty. A contrast counts only when both sides
are correct. Counterfactual repeats do not inflate the number of families.

A separate adaptive audit replaces record IDs with opaque identifiers and
shuffles display positions in two layouts per case. This addresses a concrete
design flaw: the first layouts reuse the same ID for some logical roles.
The follow-up compares the model with simple position, ID-order and uniform
ranking. It reruns identical facts under changed labels and positions,
without refitting the model. It uses the predeclared primary name.

## Results

### Verifying a decision can still leave a pathway for name influence

The same evidence and checker produced different outcomes under different
names when the proposal scores came from native Qwen. At the two-proposal
budget, **11 of 96** conditions changed between a definitive decision and
review. Three were legitimately confirmable cases and eight were legitimately
contradictable. The checker evaluated each proposed record correctly.
The discrepancy entered through the ranked proposals, which could put the
right record outside the verification budget.

With the activation gate, **0 of 96** conditions changed outcome. All
**384 of 384** planned case-name requests were admitted. All **96 of 96**
conditions had identical five-letter logits and identical checked report
bytes across the four names. All **2,976** direct comparisons matched bit
for bit, spanning 32 conditions, three name comparisons each, and 31
aligned states including the full vocabulary. All **3,840** checks of
zero identity rows passed. Native logits varied across names in 96 of 96
conditions, and native top-letter proposals changed in nine.

One example is the synthetic wildlife transport handling permit. With the
native proposals and the checker, one of four names received a confirmed
result and the other three went to review. With protected proposals, all
four received the same confirmation and record ID. Holding the name fixed
and changing the record to invalid produced a verified contradiction.
This example illustrates the mechanism. The denominator for the study
remains the entire predeclared test.

### Correctness and coverage

| Method | Definitive cases covered (48) | Correct reviews (48) | Wrong definitive reports | Different reports across names (96) |
|---|---:|---:|---:|---:|
| Native original top-letter proposal and checker | 3 | 48 | 0 | — |
| Protected original top-letter proposal and checker | 3 | 48 | 0 | 0 |
| Native top two actual record proposals and checker | 33 | 48 | 0 | 11 |
| Protected top two actual record proposals and checker | 32 | 48 | 0 | 0 |
| Protected top three actual record proposals and checker | 38 | 48 | 0 | 0 |
| Complete registry evaluation | 48 | 48 | 0 | 0 by construction |

The utility column uses one designated name per condition. The final column
uses all four. An em dash marks an endpoint omitted for that arm. Separately,
interpreting the original protected top letter without checking record
applicability produced six wrong definitive reports, all false confirmations.
The shorter protected instruction confirmed or contradicted none of the
verifiable cases after checking.

The protected two-proposal arm met the equality and zero-wrong-report gates.
It **failed** the specified 75% definitive-coverage target with **32/48**,
and confirmed only **4/12** truly valid cases. It also issued 28 correct
contradictions. The all-case score including required referrals was 80/96.
At its primary budget, 4/12 families passed both sides of the validity flip,
4/12 passed the valid-versus-deleted contrast, 4/12 passed the holder swap,
and 3/12 passed the newer-contradiction contrast. Invoking all four choices
matched the rule-only result, as predicted for the size of the ledger.
The strongest available guarantee for a closed, fully available ledger is
therefore direct registry evaluation. The two-proposal result quantifies a
behavioral tradeoff when decisions depend on model witness selection.

### The shortcut audit changed the utility conclusion

In the initial layouts, the first two IDs in sorted order cover **48/48**
verifiable cases, even with no model. The first two displayed records cover
30/48, and uniform ranking has an expected count of 26/48. The sorted-ID
shortcut invalidates using this layout to argue that the model adds ranking
utility. The exact name and activation comparisons remain valid because
the evidence and IDs were fixed across each name substitution.

After assigning opaque IDs and changing display order in two replays of each
case, there were **96** verifiable case-layout pairs. At budget two the model
verified **62/96**, compared with **55/96** for fixed display order,
**54/96** for sorted opaque IDs, and an expected **52/96** for uniform
ranking. Its coverage advantage over display order was **7.29 percentage
points**, with a descriptive family bootstrap interval of **2.08 to
13.54 points**. It verified both rekeyed layouts for **18/48** original
verifiable cases. At three proposals, the model and fixed display order
both covered 80/96. Zero wrong definitive reports were issued under the
checker throughout the rekeyed test. This is an adaptive robustness check
on the same facts, so those pairs are not independent new cases.

## Research contribution

The contribution is the experimental demonstration that **sound verification
can leave name-dependent referral intact**. On the same frozen model, same
prompt, same records and same checker, the protected execution eliminates the
observed name-dependent report switches and removes the designated input from
the measured scoring activations. The checker separately eliminates wrong
definitive reports relative to the supplied evidence. Neither result should
be inferred from the other.

The crossed interventions test that the two protections do not simply freeze
all decisions. Some changes in credential facts correctly changed the
verified status, and the complete ledger rule responded correctly in all
96 synthetic conditions. The problem of model proposal coverage survives
the intervention and becomes visible once definitive and review outcomes
are measured separately. The ID rekeying strengthens the audit of that
remaining failure. The contribution is an applied result and a reproducible
design, using established noninterference, evidence verification and
contrast-set principles.

## Limits and what each check establishes

| Problem | Evidence or control | Remaining scope |
|---|---|---|
| Equal score from a constant or always-review model | Coverage threshold plus within-family fact reversals | Designed credential rule and finite synthetic families |
| Wrong positive based on a real but irrelevant record | Check holder, exact credential, date, authority and all competing records | Trusted fields and complete supplied snapshot |
| Model leaves out a newer contradiction | Checker independently scans and compares every relevant ledger entry | Cannot discover a revocation omitted from the supplied source |
| Policy changes outcome indirectly through name-dependent proposal order | Compare native and protected logits, proposals, and report bytes on the same prompt | Proof starts with valid span review and equal retained token sequence |
| Rejection can hide unfairness | Include every planned case/name in the exact outcome gate | Broader natural layouts and formatting policy still need a representative coverage audit |
| Position or record-ID shortcut | Rekey every case twice and compare strong trivial rankers | Limited permutations and familiar English credential grammar |
| Misleading independence | Separate development and held-out families, use family-level summaries | Shared generator and one checkpoint constrain external validity |
| Incomplete or biased qualification criteria | Explicitly restrict to one simulated credential predicate | Human job-domain validation and real task labels are still needed |
| Apparent explanation faithfulness | Serialized result lists actual controlling IDs and reasons | It does not expose model reasoning or support arbitrary prose explanations |
| Unchanged general ability | Check the frozen checkpoint after inference and reuse prior clean-path benchmark context | This study does not retest general reasoning or other generation settings |
| Novelty over known work | Primary-source comparison and matched deterministic baseline | Requires independent novelty assessment and scientific replication |

The practical trust boundary is the authoritative registry and the approved
name-versus-evidence annotation. Treating ordinary résumé prose as verified
credential evidence would fall outside these guarantees. Other features can
still act as racial proxies. These experiments measure name-string influence
directly, without assigning racial identities to the people suggested by names.
The earlier checkpoint, deterministic reference, successes, and failures
remain available for independent inspection.
