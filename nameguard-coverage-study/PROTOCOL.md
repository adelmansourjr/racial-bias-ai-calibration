# Coverage recovery protocol

## Question and success criteria

The prior two-proposal gate verified 32 of 48 decisive synthetic credential
conditions. The missing cases had a valid authoritative decision in the
supplied registry whose controlling record went unproposed. The follow-up
tests whether ordering by source applicability and effective date recovers
those cases without trading away verified correctness or name noninterference.

Before test inference, freeze the generator, the checker, all proposal arms,
and the measurement script. The method passes jointly if it:

1. Verifies all decisive test cases, exceeds the original two-proposal arm on
   the same cases, and issues zero incorrect definitive reports.
2. Correctly refers all unresolved test cases and matches the fixture IDs on
   both definitive certificates and conflict reports.
3. Admits every test name, with bit-exact returned letter logits and identical
   released report bytes across names for every condition.
4. Matches every planned direct activation comparison. Record the total and
   any missing observations. Rejected requests count against the gates.

## Frozen design

The development split has three credential families and two names. The test
split has twelve disjoint credential families, eight conditions per family
and four other names. The ID generator uses independent seeds by family and
condition. Each case has four records and receives a fresh shuffled set of IDs
and a shuffled display order. The controlling record of the four decisive
conditions occupies each position once per family. The expected condition
labels are written directly from the generator edits.

For every name and case, run the same original witness prompt on frozen Qwen
2.5 1.5B, with the applicant name reaching the original tokenizer and embedding.
After protecting the name activations, extract all five answer logits. Compare
these four deterministic policies applied to the shared score vector:

- Original model ranking, top two record proposals.
- Model ranking among applicable authoritative records, top two proposals.
- Latest effective applicable record first, with protected model rank as the
  tie-break, top one proposal.
- Complete registry rule, the deterministic control. It also runs without
  model logits as an independent rule comparator.

Every model arm uses the same full-ledger checker. Missing or conflicting
evidence goes to review before proposal selection. Baseline ties use the
original answer-letter rule. Final reports are formatted by the checker.
Complete registry evaluation already scans all records, so the proposal budget
does not measure scan cost, latency, or calls to an external record source.

The primary utility unit is one case under a predeclared name. Counterfactual
names are equality checks, not extra outcomes. Report the paired rescue
and loss counts and descriptive intervals resampling twelve credential families.
Report the separate fixed-ID and display-order baselines and the uniform
ranking expectation. Audit full aligned activation arrays on the valid and
same-date-conflict conditions of every third test family, and audit all
five returned logits on every name variant.

## Why the specific fix should work

For trusted complete registry records, the set of applicable authoritative
records and its latest date come from structured record fields. If the
correct registry decision is definitive, that latest set is nonempty and
unanimous. Any record chosen from it is a valid witness. Consequently the
one-proposal policy always proposes a controlling record, and its checked
decision equals the registry rule. If the latest set conflicts or is empty,
the checker refers the case independently of model scores. This reasoning
does not make the LLM better at extracting facts from unreviewed text.
The trial should make that boundary explicit and measure the learned
ranking and applicability-only ablation separately.
