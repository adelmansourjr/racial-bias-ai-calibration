# Authenticated Name-Channel Noninterference in Frozen Language-Model Scoring

Adel Mansour | Research prototype and preprint draft | 3 November 2026

## Abstract

Matched-name tests can detect changes in language-model hiring scores, but finite
tests and learned suppression do not establish universal name invariance. We
study a narrower information-flow property: changing explicitly designated
identity text must not alter a deterministic public score when retained evidence
is fixed. We harden an internal post-embedding gate with authenticated immutable
review records, segmentation-independent scalarwise tokenization, a score-only
interface, model-integrity checks and serialized inference. The resulting
theorem is conditional on correct role labels and trusted runtime semantics;
it does not require accurate CV grading. Two abstract symbolic gather
specifications are unsatisfiable when asked for an identity-dependent retained
coordinate, while two deliberately unsafe specifications yield countermodels.
On one frozen affine-four-bit Qwen2.5-1.5B checkpoint, 108 synthetic name and
segmentation interventions produced zero score differences and 3,255 directly
equal public tensor comparisons. Five signed-request tampering variants and
runtime drift controls were rejected. Scalarwise encoding increased evidence
token counts by approximately 4.5-4.7 times on the three examples. This is a
reproducible engineering artifact and preprint draft, not a universal semantic
name-discovery system, a fairness certificate for race or a validated hiring tool.

## 1. Problem and precise contribution

The desired intervention changes only an applicant's name and leaves the
documentary qualifications and job fixed. A score that changes under that
intervention exhibits name sensitivity. That sensitivity is distinct from
population racial discrimination: names are imperfect cues, and other retained
features and policies can cause inequity. This work targets the name channel,
not the whole sociotechnical hiring process.

There are three different claims. First, a finite experiment can find no
name-caused decision changes. Second, a representation probe can fail to recover
name identity. Third, an architecture can establish that designated identity
inputs have no path to a public output. Neither of the first two alone proves
the third. For example, the hidden vector H=(qualification, name) supports a
head that reads only qualification; equal outputs do not erase the name from H.

Earlier local learned adapters reduced name-induced decision flips on synthetic
CVs while internal name-ID probes remained successful. The present study does
not reinterpret those empirical outcomes as a universal theorem. It introduces
a separate structural information-flow mode and hardens its boundary. The
contribution is an auditable system construction, failure-focused evaluation
and explicit claim ledger, rather than a claim of a novel concept-erasure
algorithm or literature-wide research novelty.

## 2. Related distinctions

Counterfactual fairness [1] motivates holding the individual's other modeled
factors fixed while intervening on a sensitive attribute. Our typed-name
intervention is narrower and does not instantiate a complete structural causal
model of race. Information-flow systems separate confidentiality from integrity
and explicitly identify a trusted computing base [2]. We apply that distinction
to exclusion of identity activations and authentication of reviewed role labels.

LEACE [3] establishes a linear guardedness property while minimizing particular
representation changes. It does not establish that this nonlinear program has
forgotten every name or association. LoRA [4] and quantized-base adaptation [5]
can alter internal computations with frozen base weights; finite adapter
training does not by itself prove all-name invariance. Concept bottlenecks [6]
can restrict decision representations to supervised concepts, but a learned
extractor can still be name-sensitive. Our current gate removes the designated
identity channel before attention instead of inferring successful erasure from
a finite concept/probe result.

## 3. Threat model

An untrusted requester controls serialized request bytes and may attempt to
change reviewed roles, text, reviewer identifiers or policy versions; insert
unusual names; fragment evidence fields; exceed limits; or overlap requests.
A trusted reviewer/operator supplies role judgments, a reviewer-key map and a
checkpoint manifest. The host process, Python/MLX implementation, deterministic
kernels and original tokenizer are trusted. Malicious code already executing
in the host can inspect buffers or keys and is not contained by Python naming
conventions.

Every identity occurrence to be varied must have the identity role, including
aliases, contacts and repeated references. Other text is evidence. A signature
cannot prove those semantic labels correct. A valid reviewer can make an
incorrect judgment; the tests preserve this negative control explicitly.
Errors, timing, admission decisions, memory use, logs and private research
observations are outside the public-score confidentiality claim. Thus the
claim is accepted-input output noninterference, not termination-sensitive or
constant-time secrecy [2].

## 4. Construction

### 4.1 Authenticated immutable review

A review payload contains the policy version, reviewer identifier, original
source SHA-256 and ordered role/text nodes. HMAC-SHA256 authenticates its
canonical JSON encoding, using at least 32 bytes of random secret key material.
Verification uses constant-time digest comparison [7]. Duplicate JSON keys,
extra schema fields, invalid Unicode and unsupported sizes are rejected.
Accepted requests are immutable bytes; verified nodes are frozen tuples, so
later caller mutation cannot alter the reviewed computation.

HMAC is shared-secret authentication, not asymmetric non-repudiation. The
verifier holds the key and can issue reviews itself. The trusted review step
must not be exposed as an applicant-accessible signing service. Experimental
keys were randomly generated in memory and not saved; public CPU-test keys are
fixtures and are not operational credentials.

### 4.2 Segmentation-independent encoding

The earlier fieldwise BPE codec did not preserve the same retained token sequence
when evidence field boundaries changed. Four actual-tokenizer examples reproduce
that limitation. The revised codec encodes each original Unicode scalar c
independently with the original tokenizer t, defining C(s) as the concatenation
of t(c) over s. It therefore satisfies C(a+b)=C(a)||C(b).

All original identity scalars are encoded as well. No anonymous replacement
string is constructed. This scalarwise convention differs from ordinary native
whole-string BPE and may affect model utility; its cost is measured rather than
hidden under the phrase 'base weights unchanged.'

### 4.3 Internal exclusion and runtime

The model computes the original embeddings for all actual source tokens,
including identity tokens. An internal gather retains only evidence coordinates
in their original order. Only that compact tensor enters the transformer;
causal masks and positions start from the compact sequence. No identity-derived
cache, residual, padding or position survives into scoring. The original 28
transformer blocks, normalization and tied vocabulary projection remain unchanged.
The scalar score is the A-versus-B logit margin and its constrained sigmoid.

The supported factory returns only an immutable two-field public score. Research
tensor observations use a trusted backend subclass, not an untrusted audit flag.
The runtime rejects external caches, checks the pinned architecture and parameter
digest, and compares module/parameter-object identities before and after inference.
A process-wide nonblocking lock serializes protected model calls across service
objects. A backend failure disables the service until rebuilding; final closure
rehashes model parameters and checkpoint files. Stable integer slots support
name-independent tie breaking, conditional on name-independent slot assignment.

## 5. Conditional noninterference theorem

Let D be an accepted authenticated typed document and Q(D) its concatenated
Unicode evidence text. Pointwise embedding lookup and the internal gather give:

    Z_0(D) = Emb(C(Q(D))).

For any D and D' with Q(D)=Q(D'), the compact input tensors are equal, including
their lengths. Each deterministic layer receives the same tensor, mask,
positions and fixed weights. Induction yields Z_l(D)=Z_l(D') at every public
layer; the fixed readout then yields PublicScore(D)=PublicScore(D'). No grading
accuracy premise appears. The proof covers every accepted identity value and
evidence resegmentation under the codec, not just the tested name list.

The HMAC adds a separate computational authenticity premise under its standard
security assumptions. It does not strengthen the semantic completeness premise.
Names left in evidence are outside the theorem. Conditionally, the downstream
state is independent of the designated identity intervention given Q. Real
population correlations between names and retained qualifications need not
vanish, and other racial proxies remain possible.

The one-block and two-block SMT specifications use arbitrary arrays, arbitrary
deterministic embedding coordinates, insertion positions and identity lengths.
Z3 finds no differing retained coordinate (UNSAT). General finite-list extension
and transformer induction are handwritten arguments. The solver does not verify
Python, tokenizer behavior, floating-point kernels or role judgments end to end.
An unsafe original-position specification admits a name-length countermodel;
a finite-testing specification admits an untested-name failure (both SAT).

## 6. Experimental design

The frozen GPU plan was saved before inference and records source hashes,
inputs, variants and acceptance checks. The design crosses three generated
evidence bodies, 12 identity strings and three evidence/identity segmentation
variants: 108 forwards. Strings include Arabic, Chinese, Devanagari and Korean,
full-width letters, zero-width/NUL characters, an instruction-like string, a
skill/name collision and a 1,000-character field. Names are synthetic treatment
strings, not assigned race labels or real applicant records.

For each fixed evidence body, direct NumPy array comparisons check the compact
embedding, all 28 block outputs, final normalization and full final-position
vocabulary logits: 31 public tensor stages per forward. These comparisons do
not substitute matching hashes for array equality. A separate check compares
the manual block path with the original Qwen path on the same scalarwise
evidence. Scalar scores must agree exactly across variants and remain
nonconstant across evidence bodies.

Five request mutations test source, role, reviewer, policy and source-hash
authentication. Actual runtime controls test overlay rejection, graph drift,
serialization, failure poisoning and final integrity. Thirty CPU tests include
2,000 seeded random Unicode insertion/resegmentation cases. This is deterministic
fuzzing, not an exhaustive Unicode proof or a representative sampling study.

The model is one pinned affine-four-bit Qwen2.5-1.5B-Instruct runtime on Apple
Silicon MLX. It is distinct in precision from the original BF16 download.
No adapter is trained or installed in this hardening study. Clef and unrelated
GPU training jobs are not used or modified. The exact model parameter digest,
dependencies, run plan and code are in the artifact.

## 7. Results

All 108 forwards passed. Three forwards established per-body reference tensors;
105 counterfactual forwards yielded 3,255 directly equal public-stage comparisons
within fixed-evidence groups and a maximum name/segmentation-induced score
difference of zero. Different evidence bodies produced different margins:
0.1602668762, -0.0963649750 and -0.3481140137. This rejects a constant-only
demonstration but does not establish grading correctness.

| Measure | Result |
|---|---:|
| Actual named-input forwards | 108 |
| Direct public tensor comparisons | 3,255, all equal |
| Maximum matched score gap | 0 |
| Signed-payload tampering variants rejected | 5/5 |
| Real-tokenizer BPE segmentation failures reproduced | 4 |
| Revised codec segmentation checks | all equal |
| CPU tests | 30 passed |
| Seeded Unicode property cases | 2,000 passed |
| Safe SMT specifications | 2 UNSAT |
| Unsafe SMT specifications | 2 SAT |

The runtime smoke test passed the final factory path, process-wide busy
rejection, refusal to reuse a poisoned service and module-drift detection.
Final parameter-content and checkpoint-file checks passed. No reviewer key
was saved. Ordinary BPE required 59, 60 and 61 evidence tokens, compared with
271, 281 and 274 scalarwise tokens: 4.59x, 4.68x and 4.49x, respectively.

These counts are conformance evidence, not independent applicant trials. We do
not attach a binomial population confidence interval to repeated forwards.
The CPU result verifier reconstructs source hashes, factorial coverage and
records; it does not independently re-run GPU inference or reconstruct omitted
large tensors. Both this distinction and the abstract solver boundary are part
of the reported result.

## 8. Limitations and publication status

The largest unresolved limitation is semantic identity-role correctness. A name
may be an applicant, a client, a software identifier or a legitimate fact. A
text-only program cannot guarantee two different correct roles for identical
observable strings in different latent contexts. This does not prove that all
learned invariant functions are impossible; it identifies an information gap
the present system does not solve.

Signed incorrect reviews still permit identity leakage through evidence.
Timing, codec caches, request lengths, rejection behavior and malicious host
code are outside the output theorem. Graph/object checks and final hashes are
not hardware attestation. Scalarwise encoding is token-expensive and its
grading accuracy is unvalidated. External slot assignment or an application
that breaks ties by name can invalidate an end-to-end ranking claim.

Only three synthetic evidence bodies and one model/runtime were studied. No
independent reviewer study, external replication, real applicant-outcome study,
multi-model replication or claim of new academic theory is supplied. The
system is not ready for consequential hiring deployment. This manuscript is a
preprint draft suitable for transparent portfolio presentation and further
review, not a peer-reviewed publication or evidence that all racial bias has
been eliminated.

## 9. Reproducibility and next experiments

The package contains source, frozen plans, solver specifications and outputs,
complete synthetic intervention records, a read-only verifier, tests, a
source/result manifest and a CPU CI recipe. Model weights and reviewer keys
are excluded. The exact pin intentionally rejects other checkpoints. A new
GPU run must use a new output copy rather than overwriting completed plans.

The next substantive studies should measure human role-annotation agreement
and omitted-name error rates on separately consented CVs; replicate the
structural conformance with independent code and another tokenizer/model;
study more efficient segmentation-invariant codecs; and measure the utility,
latency and admission costs. These are future work, not completed evidence.

## Data, ethics and assistance disclosure

Only generated CV-like text is released in this extension. No race labels are
inferred from individual names. The original project used public résumé/job
datasets, but those records are not imported into this study or redistributed.
Publishing this code is not a hiring deployment. Code, prose and analysis were
developed with AI coding assistance; the artifact should be independently
reviewed before academic submission. Model and dependency terms remain with
their respective owners. The draft has no DOI or peer-review status.

## References

1. Kusner, Loftus, Russell and Silva. Counterfactual Fairness. 2017.
   https://arxiv.org/abs/1703.06856
2. Chong, Vikram and Myers. SIF: Enforcing Confidentiality and Integrity in Web
   Applications. USENIX Security, 2007.
   https://www.cs.cornell.edu/andru/papers/usenix07-html/paper.html
3. Belrose et al. LEACE: Perfect Linear Concept Erasure in Closed Form. 2023.
   https://arxiv.org/abs/2306.03819
4. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.
   https://arxiv.org/abs/2106.09685
5. Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023.
   https://arxiv.org/abs/2305.14314
6. Koh et al. Concept Bottleneck Models. 2020.
   https://arxiv.org/abs/2007.04612
7. Python Software Foundation. hmac - Keyed-Hashing for Message Authentication.
   https://docs.python.org/3/library/hmac.html
