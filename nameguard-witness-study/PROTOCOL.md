# Credential witness study protocol

## Research question and primary claims

This follow-up targets a specific open failure from the utility study: the
improved name-isolated readout falsely confirmed 12 of 12 missing-evidence
cases. An output letter expressed a judgment without identifying an admissible
supporting fact. We test a prompt that asks for a record ID and a checker that
can validate the proposed witness against a complete supplied registry.

A positive result requires all three of the following:

1. Exact equality of protected witness logits and checked explanation bytes
   across all four planned names per held-out CV condition. Report both
   admission and equality, and count a rejected variant against this gate.
2. Zero incorrect definitive reports under the verifier. Count all unsafe
   confirmations and unsafe contradictions and report valid confirmations
   separately so the procedure cannot pass by referring every case.
3. At least half of verifiable cases produce a definitive checked report.
   Count a wrong or rejected proposal as a lost definitive case. Report
   exact evidence-ID matching and response to status, holder and date edits.

This is a deterministic conformance claim for the stated source and runtime,
and a finite-sample utility check. These thresholds are frozen locally before
held-out inference. A negative outcome remains a result.

## Design

- One pinned affine-four-bit Qwen2.5-1.5B-Instruct on Apple MLX.
- Three development credential families and two names. Twelve disjoint test
  families, each with eight interventions, across three surface templates.
  Test cases are authored synthetic records with known facts, not real
  hiring outcomes. Name changes leave the evidence and record order fixed.
- The original named prompt reaches the tokenizer and embedding. A reviewed
  role contract is frozen from a distinct fixed name panel. The inherited
  gate excludes the identity activations from protected blocks.
- Four planned names cross every test condition. Record order depends on
  family and stays fixed under within-family edits. The options list only
  source references. The evaluator never uses the fixture label to check
  a proposal. The explicit fixture oracle is based on the intervention.
- Primary GPU comparisons: native and protected with the **same new prompt**
  under all four names. This isolates what the activation gate changes.
  A short pointer-only instruction on the protected path removes the rubric
  and examples jointly on the primary name. The two prompt effects are not
  identified independently. Checked and unchecked reports use the very
  same logged model proposals. A deterministic registry-only comparator
  tests whether the model adds coverage under this closed contract.
- For four predeclared families (0, 3, 6, 9), compare all 31 aligned protected
  arrays against the first name's arrays by bit pattern. Check identity
  rows are zero at each non-vocabulary stage. Measure stable explanations
  under name changes and require the correct visible decision change for
  status, holder, and date edits to be counted as pair successes.
- Count both definitive and review decisions. Report wrong definitive calls
  divided by all cases and coverage among genuinely verifiable cases.
  Name variants are exact conformance repeats. Utility uses a single
  predeclared name. The twelve credential families are the bootstrap unit.
  Report descriptive intervals and all contrast kinds without claiming
  population-level estimates.

## Oracle and trust boundary

The checker requires a complete supplied authoritative ledger with correct
holder, credential identity, effective dates and validity. Drawn from that
ledger, each model-visible line is rendered deterministically. A newer
contradiction or conflict within the supplied ledger cannot be hidden by
the model omitting a citation. Absence from the supplied ledger is a reason
for review, not a statement about real-world credential status. Recomputing
the rule from the ledger needs no model. That comparator prevents crediting
the LLM with the checker's safety guarantee.

This protocol does not establish authority or completeness for arbitrary
résumé prose, correct annotation without a reviewer, demographic parity,
external-lab replication, timing noninterference, or suitability for hiring.
