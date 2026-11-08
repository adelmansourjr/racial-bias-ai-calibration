# Claim, proof, and evaluation protocol

## 1. The property being established

Let an admitted request have native token IDs t, reviewed evidence positions P,
and evidence sequence e = t[P]. The source and complete original token IDs
reach the model unchanged. Identity denotes the reviewed applicant-identity
role, rather than a dictionary of known names.

For a fixed checkpoint, deterministic inference runtime and score readout, the
claim is:

~~~text
For every admitted x,y with e(x) = e(y): score(x) = score(y).
~~~

This quantifies over admitted token sequences, including previously unseen
names. It does not quantify over every arbitrary raw document: requests must
have correct reviewed roles, a native token-boundary witness, a final evidence
readout position and at most 2,048 original tokens. The exact evidence contract
compares immutable token tuples directly, rather than treating matching hashes
as a proof of equality. Hashes identify artifacts and summarize measurements.

The API exposes a scalar margin and its logistic transformation. This experiment
does not implement or certify arbitrary autoregressive text generation.

## 2. Internal activation construction

Let E be the frozen token embedding lookup. G_P gathers the ordered evidence
rows and S_P scatters them into full original slot geometry, filling identity
positions with zero. Let B_l be native transformer block l, executed without a
KV cache on the gathered evidence sequence. Its causal mask and RoPE positions
are constructed from that evidence sequence. Define:

~~~text
A_0 = E(e)
A_(l+1) = B_l(A_l)
H_l = S_P(A_l)
G_P(S_P(A)) = A
~~~

The implementation first evaluates E(t) on the complete original named input,
zeros identity slots, then gathers and scatters at every block. It preserves
original tokenizer IDs and full stored slot count. The active block schedule
uses evidence length, so it differs from ordinary full-length dense attention.
This is an internal structural intervention implemented in inference code. It
is not a successful learned QLoRA forgetting result or a guarantee possessed by
the unchanged base checkpoint running without this execution path.

If e(x)=e(y), their evidence embeddings coincide. By induction, each native block
receives the same values, shape, mask and logical positions. Deterministic block
execution therefore gives equal A_l, final normalized evidence states, final
vocabulary arrays, and score. Different physical layouts H_l need not have the
same shape; their gathered evidence states coincide and identity rows are zero.
This avoids the observed length-dependent dense-kernel numerical differences.

A fixed calibration m' = m - c preserves equality because c is independent of
the request name. Any subsequent tie-breaking, shortlist policy or metadata must
also depend only on protected outputs and name-independent candidate IDs to
inherit the property. This revision directly checks scoring, not a new ranked
hiring deployment.

## 3. What the proof cannot establish

A hidden state H=(qualification,name), with output f(H)=qualification, supplies
a counterexample to the inference that equal outputs imply absent name
representations. Behavioral parity alone does not prove activation erasure.
Here, the scoring branch has equal aligned evidence states and zeroed identity
slots after the boundary. The earlier original embeddings still encode names.
Private token length, addresses, timing and admission behavior can still differ.

Subtractive LoRA can remove a selected low-rank direction. If a projection P
removes a fitted difference subspace U, then P u = 0 for u in U. This says nothing
about unseen differences outside U, nonlinear reconstruction, retained side
paths or changed positional computations. Earlier learned pilots in this
repository demonstrate residual score variation. Exact isolation here follows
from the structural recurrence, rather than a claim that fitted directions span
every possible name feature.

Removing every racial proxy while preserving every qualification decision is a
separate, stronger goal. Evidence can correlate with demographic variables, and
job requirements can themselves be unfair. The present property excludes the
designated identity channel; it does not adjudicate those policies.

## 4. Experimental design and preserved controls

The dated local plan freezes the four main experiment files and inherited
runtime dependency hashes before inference. It is not an externally registered
preregistration. The same 192 earlier named CV/job inputs are replayed through
score(), comparing the previous scores and 31 aligned stage fingerprints per
input. Additional checks verify unchanged original IDs, zero identity slots,
model parameter digest and checkpoint files. The old records remain local.

For task calibration, eight synthetic credential types provide 32 fitting rows.
A threshold minimizes fitting classification errors, then absolute shift, then
signed threshold within a finite candidate grid: zero, one unit outside each
extreme, and adjacent distinct-margin midpoints. The tie-break is over this grid,
not a continuous minimum-shift optimization. The selected artifact is saved
before test calls. Six other
credential types, two prompt styles, two qualification states and four names
provide 96 held-out rows and 24 same-evidence condition groups. Credential
patterns revisit an earlier project rubric; this is not an untouched benchmark
or independent hiring-label validation. Repeated names are conformance checks,
not 96 independent applicants.

The main clean control compares all vocabulary logits on the same 32 ARC-style
inputs. This establishes preservation on those inputs. It does not constitute a
full benchmark or a statistical general-capability noninferiority result.

The tokenizer sweep contains 120 synthetic cases in five layouts. Its follow-up
first checks native token compatibility, then exact equivalence to the first
approved evidence sequence in each layout. It reports 103 matches, nine
additional token-compatible cases rejected for changed evidence, and eight
token-boundary failures. Anchor choice and review policy affect those counts.
The final contract is exercised on the GPU with two qualification conditions,
short/long names, and stale-evidence rejections.

## 5. Operational safeguards and trust boundary

Authenticated role packets are immutable bytes. Both the evidence wrapper and
scorer validate them. Two configured credential signatures bind the same source
and roles; signatures alone do not prove independent human review or correct
semantics. The CLI creates temporary fixture keys for a local demonstration.
Real reviewers or a separately validated structured input pipeline must supply
role truth in a real study.

Malformed requests and incompatible evidence reject before inference and remain
recoverable. Backend/model-guard failures poison the service, remove exception
details from public errors and require a fresh service. Request and global model
locks reject overlapping work. Close is idempotent. The host, Python process,
review authority, selected checkpoint and operator configuration are trusted.
This is not a sandbox against a malicious host that directly reads model memory
or invokes private diagnostic methods.

## 6. Research context

Zhao et al. (2021) identify answer-label priors and contextual calibration. Our
labeled synthetic threshold fit is distinct from their content-free method:
[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html).

Sabelfeld and Myers (2003) frame confidentiality through information flow and
noninterference. The protected score path is the observer in our scoped claim:
[Language-Based Information-Flow Security](https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf).

Russinovich (2026) studies marked-span vetoes and bit-identical outside
activations under same-length substitutions. This is a direct antecedent, not
an unrelated comparison. The present implementation additionally addresses our
observed variable-length numerical-geometry failure through gathered native
blocks: [What the Model Cannot See](https://arxiv.org/abs/2610.15883).

The repository's earlier [31-source review](../nameguard-validation/literature/)
and [novelty assessment](../nameguard-validation/literature/NOVELTY-ASSESSMENT.md)
remain relevant. This revision is a hardened implementation and measured
follow-up, with no claim that name-blind routing or information-flow isolation
is a newly invented general principle.

## 7. Reproduction and release standard

Public CI runs CPU tests and artifact consistency checks. Actual inference
requires the separately acquired pinned checkpoint and recorded MLX environment.
Private real CV/job records are not included in the release. Public verification
therefore audits the released records and source integrity; it cannot constitute
independent replication of those private GPU measurements. Synthetic boundary
and integration checks are runnable with the checkpoint.

A stronger external study would independently annotate identities and retained
roles, obtain job-rubric labels, predeclare equivalence tolerances and power at
the CV/job unit, test untouched CVs and prompt families, and separately audit
admission rates and demographic outcomes. These steps require evidence beyond
another iteration on the existing private sample. The release supports a
technical-report/preprint claim with explicit premises, not a universal racial
fairness or hiring-accuracy certificate.
