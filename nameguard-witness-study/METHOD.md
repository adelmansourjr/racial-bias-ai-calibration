# Proofs and comparison design

## Exact name noninterference at the model boundary

Let `S` denote the reviewed source, `N` the designated applicant-name field,
and `E(S)` the exact tuple of evidence token IDs retained by the protected
path. The raw name remains in the native tokenizer input and original
embedding lookup. The gate zeros the designated activations and computes
each frozen native block on the gathered evidence rows.

For each admitted source, the layer recurrence on observed rows is

\[
H_0 = \operatorname{Embed}(E(S)),\qquad H_{l+1}=B_l(H_l).
\]

The blocks use the same evidence-relative positions, mask, parameters, and
deterministic execution configuration. If two reviewed inputs admit the
same tuple, their initial rows are equal. Induction gives equal aligned
states and the same vocabulary vector through the final readout. Scatter
back into the physical slots gives zero identity rows. The proof compares
aligned evidence states, while original identity tokens and their slot
counts can differ before the boundary.

The witness rank uses only protected logits, fixed record IDs, a stable
tie-break, and the evidence present in the source. The checker uses the
same supplied registry and policy. Composing deterministic functions
therefore yields byte-equal reports under the same admission and retained
evidence assumptions. This result covers the serialized decision interface.
It treats rejection, timing, operational load and initial raw embeddings
as separate observation surfaces. We include admission in the empirical
gate instead of restricting the denominator to successfully scored cases.

## Why model confidence alone cannot establish a credential

For a supplied registry \(D\), required credential \(q\) and cutoff \(t\), let
\(A(D,q,t)\) be records from the authoritative registry for the applicant
with the exact credential and effective date at most \(t\). If this set
is empty, the policy result is review. Otherwise let \(L(D,q,t)\) be its
subset at the maximum date. If the records in \(L\) conflict, the result is
review. A unanimous \(L\) defines the status. The certificate contains the
sorted record IDs that determined the result.

A model pointer can be checked locally by looking up the selected record.
That is insufficient to establish its authority over the decision.
Omitting a more recent contradictory record can make the same selected
quote incorrectly decisive. Our checker scans all supplied entries and
checks the two separate claims that the selected record is applicable and
that there is no overriding or conflicting evidence in that snapshot.

A case with an undisclosed revocation and a case with no revocation can have
identical supplied documents. No function of those documents can always recover
both real-world statuses. This identifies the completeness assumption exactly.
The experiment uses supplied synthetic registries with directly known facts.
Issuer identity, record coverage across organizations, and assignment of
the holder role are trust inputs in an external workflow.

## Soundness and budget

The proposal arm ranks the actual record IDs in the visible options.
The budget counts how many proposals can be checked, with fixed tie
ordering and no choices for nonexistent records. Pass a definitive status
only if the policy based on the complete supplied snapshot is definitive
and one of the first \(k\) record proposals lies in \(L\). The checker emits
the policy status and controlling records. It sends missing, contradictory,
or budget-exhausted evidence to review.

For every score vector, any definitive report equals the policy result
and contains the correct controlling registry IDs. This follows from
the definition and holds even if model scores are adversarially wrong.
Every unresolved case is referred regardless of the score vector.
The argument supplies no lower bound on how often a valid proposal appears.

For verifiable context \(i\), define \(r_i\) as the first rank of an ID in its
controlling set. At budget \(k\), verified definitive coverage is exactly
\(\sum_i 1[r_i\leq k]/n\). Increasing the budget cannot introduce a wrong
definitive decision or reduce coverage, as long as checker, snapshot and
ranking are held fixed. It can preserve identity dependence if the rank was
computed by the unprotected model. Identical scores and report bytes
therefore remain separately tested rather than inferred from soundness.

A larger budget nearly recovers the deterministic rule on small ledgers.
The benchmark uses at most four visible records and the registry-only
baseline scans them all. Thus the closed ledger demonstrates behavior and
a separation of guarantees. It supplies no empirical evidence of a latency,
cost, or coverage advantage over a deterministic registry implementation.
A retrieval study with incomplete candidate access would require a different
certificate for evidence completeness.

## Counterfactual tests

For each family, choose one valid evidence context and pair it with seven
different interventions. Re-evaluate with four names per condition. Each
name-substitution comparison holds source evidence, required credential and
authored record order fixed. Each evidence comparison holds the task and
primary name fixed and changes a declared fact. The status, evidence deletion,
role swapping, unrelated credential, future-only, conflict, and newer-invalid
edits are all reported.

A successful contrast requires the correct answer on both sides. Name
invariance by itself is compatible with a constant system. A zero unsafe-error
result by itself is compatible with always referring to review. The joint
result therefore includes admission, exact observed equality, the unsafe
definitive count, and useful coverage among cases whose status is established.
Source ID equality prevents a correct label with irrelevant evidence from
counting as success in the certificate endpoint.

Reranking proposals follows a development observation about the five-letter
readout. The test uses twelve disjoint credential families and three render
templates absent from development. Reports use the full denominator with one
primary name per scenario. Each scenario's name variants are conformance
runs, not independent evidence situations. Descriptive bootstrap intervals
group the eight related scenarios of each family and resample families.

## Prior work and positioning

Fact verification with supported, refuted and insufficient-evidence
decisions has clear precedents. This experiment adds within-family name
and record interventions, then evaluates label and source attribution
jointly rather than claiming to invent evidence-grounded inference.
[Thorne et al., FEVER](https://aclanthology.org/N18-1074/)

Contrast sets and tests of syntactic shortcuts likewise predate this work.
The current benchmark crosses contrast sets about credential evidence with
exact name substitution and direct transformer-state measurements.
[Gardner et al., Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/)
· [McCoy et al., HANS](https://arxiv.org/abs/1902.01007)

A model proposal followed by a verified result is an application of the
certifying algorithm pattern. The potential research contribution is the
specific crossed audit and measured separation of wrong answers, abstention
and name dependence in a frozen CV scorer. Existing span-level
noninterference and earlier project results supply the gate architecture.
[McConnell et al., Certifying Algorithms](https://doi.org/10.1016/J.COSREV.2010.09.009)
· [Earlier NameGuard study](../nameguard-peer-review/paper/manuscript.md)
