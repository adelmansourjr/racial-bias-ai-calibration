# Frozen benchmark protocol

**Question.** Can the same frozen Qwen model check source evidence under an exact
name-channel constraint, while a selective decision rule achieves a useful
risk–coverage point on held-out human annotations?

**Unit.** One candidate claim per question. The public SQuAD 2.0 development release
supplies both labels and source offsets. In each included paragraph, hash-based
selection picks one answerable question with an exact gold answer and one officially
unanswerable question with an exact, human-written plausible answer. Both quotes
literally occur in the paragraph. Keep contexts of at most 1,600 characters and
reject embedded chat delimiters. The claim is whether the excerpt supports the
proposed answer to the question. Labels do not appear in model inputs.

**Partition.** Hash-order 35 whole articles before any inference. Take eight for
development (72 paragraph pairs), 17 for the held-out evaluation (156 pairs), and
reserve ten unused articles. Order pairs and candidate questions by fixed hashes.
The name intervention changes only the formally marked applicant-name span.
Two separate fixture names compile evidence contracts. The primary inference cohort
uses one name. A hash-fixed 20-case conformance subset uses all four evaluation
names and both A/B mappings. Check original token IDs, bitwise public activations
at every recorded stage, logits, deterministic decision, and report bytes.
The matched subset includes a native outcome comparison.

**Rule.** For each case, ask for A or B twice with complementary support mappings.
Three predeclared statistics are the single forward margin, the average sign-corrected
margin, and the minimum sign-corrected margin. For each statistic, pick the lowest
development-score cutoff with at least 12 accepts and at most 4% empirical error,
keeping score ties together. For the primary rule, select the statistic accepting
the most development cases among the mean and minimum variants, preferring mean
on ties. If neither qualifies, refer all. Freeze the threshold before holdout model
inference. Apply unchanged single- and two-mapping cutoffs on holdout for comparison.
The literal-span comparator accepts every case. Record observed coverage, false
support among accepted decisions, and AUC. Unadmitted claims are automatic referrals.
No threshold search on holdout changes the primary result.

**Audit.** Freeze an independently randomized permutation of all holdout IDs before
inference. After freezing holdout decisions, take the first 80% of accepted cases
in that order (rounded up, at most 160). Invert the one-sided hypergeometric test at 5%
for an upper bound on the finite accepted-set error fraction. Compute the exact
census afterward as a cross-check. The bound is design-based conditional on the realized finite
accepted set and a uniform permutation independent of decisions. The bound has no future-population
interpretation. Name substitutions are repeated measurements and never increase
the audit sample size.

**Success gate.** Predeclare exact conformance on all 20 protected name orbits,
at least 20% accepted of the 312 held-out cases, and a one-sided 95% upper audit
bound of at most 10% error among accepted claims. Census results and the native
comparison are descriptive checks. Reserve articles cannot rescue a failed primary
gate and may only be used for separately frozen future analyses.

**Scope.** This compares binary support for a supplied quote on human-annotated
Wikipedia text. The public dataset may overlap pretraining. It does not measure
evidence discovery, general citation entailment, real hiring decision quality,
all sources of name leakage, or absence of identity representations throughout
the model. Earlier real-CV evidence is reported separately.
