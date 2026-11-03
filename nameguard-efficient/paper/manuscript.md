# Native-Token-Count Name-Channel Noninterference with Authenticated Role Review

Adel Mansour | Research prototype and preprint draft | 3 November 2026

## Abstract

Structural name-channel isolation can make model scores invariant to explicitly
designated identity fields, but a segmentation-independent scalarwise codec can
inflate transformer sequence length. We derive and implement a canonical-evidence
factorization that uses exactly the original tokenizer's token count for retained
evidence. The complete original named source is tokenized and embedded first;
only then is one evidence sequence constructed and retokenized inside the model
backend. This changes the computational boundary, not the underlying language
model weights. We also add source-exact span validation, registered-reference
coverage, explicit retention of same-spelled skills or third parties and two
credential endorsements of the same resolved payload. On one frozen Qwen2.5-1.5B
runtime, 108 controlled forwards yielded identical scores and 3,360 directly
equal counterfactual public-stage arrays. All six native-reference hidden and
vocabulary comparisons matched. Across 32 generated text cases, transformer
tokens fell from 25,201 to 8,457, matching native evidence BPE exactly. A local
21-pair benchmark measured median service latencies of 334.4 ms versus 81.5 ms,
with median paired speedup 4.10x. All 108 registered-omission controls and 36
review-conflict controls were rejected. Unknown aliases remain a deliberate
negative control: authenticated agreement is not a proof of semantic role
correctness. The work is a reproducible preprint draft, not universal automatic
name discovery, whole-model unlearning or a racial-fairness deployment certificate.

## 1. Scope and motivation

The requested property is name noninterference: changing an applicant identity
must not change a public score when job-relevant retained evidence stays fixed.
Accurate grading is a separate property. A score can be invariant and wrong;
a constant score is trivially invariant but provides no qualification sensitivity.
We require the demonstration to remain nonconstant across evidence bodies,
without claiming those scores establish actual hiring usefulness.

The prior local prototype embedded all original source fields, then excluded
identity activations before attention. To avoid tokenization changes when public
text was split across annotation nodes, it encoded each Unicode scalar separately.
That convention was structurally convenient but expensive: the three reference
bodies used roughly 4.5 times ordinary evidence BPE length. The present extension
asks whether strong segmentation independence actually requires that cost.

Our answer is no, if the architecture is allowed to construct a canonical
evidence sequence internally before its scoring transformer. This is not
autonomous learned name erasure. It is a new explicit information-flow boundary
with typed role metadata. The construction is related in purpose to earlier
internal source/evidence routing in the same project; we do not claim literature-
wide novelty for canonicalization or universal learned unlearning.

## 2. Related work and distinctions

Subword BPE [1] merges useful character/byte sequences and helps handle an open
vocabulary. The standard tokenization pipeline distinguishes normalization,
pre-tokenization, the subword model and post-processing [2]. Applying such a
tokenizer independently to arbitrary pieces generally does not reproduce
tokenizing their concatenation. This explains why deleting name-token rows from
a context-dependent full-source BPE sequence does not always produce native
evidence token IDs or positions.

Counterfactual fairness [3] motivates controlled changes to a sensitive factor,
but our name-channel intervention is narrower than a causal model of race.
Information-flow systems distinguish integrity of labels from confidentiality
of outputs [4]. Our review signatures address later tampering, while the
architectural theorem addresses dependence on designated identity inputs.
Neither addresses every racial proxy or fairness of the underlying job policy.

Agreement coefficients [5] describe consistency among annotations, not their
semantic correctness. Two signatures on an identical payload are authenticated
agreement by two credentials; they are not evidence of two independent human
judgments. We explicitly preserve the possibility of a shared missing alias.
This extension evaluates mechanical annotation safeguards with a synthetic role
oracle, not an invented human-review accuracy study.

## 3. Formal problem and previous encoding cost

Let D be a bounded source-exact list of identity/evidence text nodes. Let X(D)
concatenate all original text, and Q(D) concatenate evidence text only. Let T
be the fixed original deterministic tokenizer. The former scalarwise codec was:

    C_scalar(Q) = T(c1) || ... || T(ck),
    m(Q) = sum_i |T(ci)|,        n(Q) = |T(Q)|.

If a codec must satisfy C(a+b)=C(a)||C(b) for every possible scalar split,
induction forces this scalar factorization. Whole-string BPE is generally not
such a homomorphism; therefore retaining arbitrary independent-piece encoding
while also retaining every native cross-boundary merge is an incompatible
requirement in general. The overhead varies by script and text, not a universal
constant. It should be measured, not inferred from English examples alone.

Three options are distinct. Fixed-field BPE is inexpensive but makes public
field boundaries part of the equivalence contract. Scalarwise encoding permits
arbitrary splits but loses merges. Canonicalizing retained evidence before one
tokenizer call permits both arbitrary evidence segmentation and native evidence
BPE. We implement the third option while preserving actual original-source
embedding before canonicalization.

## 4. Internal canonical-evidence construction

The actual forward order is:

    R = Emb(T(X(D)))                  # all original names embedded first
    E = Q(D)                         # constructed inside model backend
    H0 = Emb(T(E))                   # native evidence token embeddings
    H_(l+1) = F_l(H_l; theta)        # original 28 blocks, no raw-source cache
    S(D) = fixed_original_readout(H_L).

Raw R is evaluated before E is constructed. No anonymous text is supplied in
place of the original source at the external encoding step. Nevertheless,
ordinary code inside the model explicitly constructs E from reviewed roles.
This fact must not be omitted when describing the method. The system has not
learned to identify and ignore every raw name without supporting metadata.

The raw branch supplies no tensor, source length, cached state or position to
the scoring branch. Original native evidence masks and positions start from
T(E). The model uses its original embeddings, transformer, normalization and
tied vocabulary head with the same FP32-input readout convention in all native
comparisons. No learned adapter is installed in this extension.

## 5. Conditional proof and efficiency derivation

For accepted documents D,D' with Q(D)=Q(D'), deterministic tokenization yields
T(Q(D))=T(Q(D')). Original evidence embeddings are equal. Each fixed transformer
layer receives the same preceding tensor, positions and mask; induction gives
equal hidden states at every scoring layer and an equal deterministic score.
The proof covers accepted new identity values and arbitrary resegmentation of
the retained evidence. It requires correct role assignments that preserve Q;
it makes no accuracy assumption and no claim about unannotated name discovery.

The scoring transformer processes exactly n(Q) positions. This is equality with
the original tokenizer on the same evidence, not a claim that greedy BPE is
globally token-minimal over every possible vocabulary segmentation. The original
source is also embedded, so total embedding visits are n(X)+n(Q). Review checks
and two tokenizer calls also cost CPU work. A name-bearing raw CV can have
different boundary merges from Q; we do not assert n(X)>=n(Q) universally.

A rough prefill scaling with L layers, width d and MLP width f is:

    Cost(n) = O(L*(n*d^2 + n*d*f + n^2*d)).

Changing m to n reduces token-linear block terms by m/n and the attention term
by (m/n)^2, while adding raw embedding and canonicalization overhead. These
ratios are analytical component comparisons, not promised end-to-end speedups.
The local latency benchmark measures the complete supported service calls.

The symbolic factorization specification asks for unequal outputs under equal
evidence with arbitrary deterministic tokenizer/embedding/transformer/readout
functions; Z3 returns UNSAT. A separate independent-piece token-count
counterexample returns SAT. These small specifications check congruence and
non-homomorphism assumptions, not program correctness or GPU semantics.

## 6. Stronger annotation integrity and omission checks

Reviewers annotate exact original source spans, preserving every input character.
The signed payload contains a policy version, ordered roles, source hash,
registered identity strings, explicit same-spelled retained meanings and a
resolved status. Exact text/offsets, overlap, registry coverage and two known
distinct credential signatures on the same payload are checked before model
execution. Duplicate JSON keys, wrong signatures, changed policies, stale spans,
unresolved proposals and conflicting payloads reject.

Registry detection supports casefolding, accents, full-width forms, combining
characters, zero-width formatting characters and punctuation-separated names.
Detection does not normalize source text used for scoring. Candidate spans
remain unresolved until semantic review. A name matching the programming
language Python or a third-party client requires a source-bound retained role
and reason rather than automatic removal.

This mechanism checks declared textual coverage, not every nickname, indirect
reference, transliteration or encoded identifier. Two credentials can be owned
by one operator or share the same semantic mistake. They need distinct keys,
but the program cannot prove two independent human judgments. The synthetic
study explicitly measures none of those human properties.

Let A mean that roles are complete/correct and the retained evidence is fixed
under the identity intervention; let B mean the structural runtime is correct.
For an output violation V, the conditional theorem gives V impossible on A and B.
Under a specified input-distribution probability model:

    P(V) <= P(A^c union B^c) <= P(A^c) + P(B^c).

Neither probability is estimated from real CVs here. Two reviewer miss rates
p1,p2 imply joint miss probability p1*p2 only with a justified independence
assumption. Correlated joint misses can be as high as min(p1,p2). We do not
claim that requiring two signatures squares the semantic error rate.

Character-level agreement, kappa and identity-character F1 are available as
diagnostics. Kappa is undefined when its expected-agreement denominator is zero;
the code returns null. High character agreement on a long CV can still conceal
a short omitted name. Future evaluation needs independent adjudicated source
truth and document-level omission/false-removal measurements [5].

## 7. Experimental design

The source-hashed plan is frozen before inference. Conformance crosses three
generated evidence bodies, 12 identity strings and three source/segmentation
variants: 108 full forwards. Names include several scripts, full-width letters,
zero-width/NUL characters, a skill-name collision, an instruction-like string
and a 1,000-character value. Every original source is embedded intact. Three
forwards establish per-body references; 105 counterfactual forwards compare
32 public stages each: canonical IDs, evidence embeddings, 28 block outputs,
final normalization and full final-position vocabulary logits. Direct array
comparisons, rather than hash equality alone, check 3,360 counterfactual stages.
Three native reference forwards separately compare final hidden states and
full vocabulary arrays, giving six native-reference comparisons.

The token-count corpus contains 32 generated cases: eight language/style
templates with four repetition lengths. It is not real applicant data or a
population sample. Native evidence BPE, old scalar public tokens, raw-source
BPE and total embedding-token visits are all recorded. No inputs are truncated.
One old/scalar example exceeds its public-token limit; all canonical cases fit.

Latency uses two warm-ups per method/body and seven paired repetitions across
three bodies (21 pairs, 42 timed calls), with shuffled order. Signing and model
load are excluded; review verification, encoding, embedding, inference and
per-call graph checks are included. The new service performs two signatures and
registry checks; the old scalar service uses its earlier single-signature API.
This is a service-package comparison, not a token-count-only ablation. Twelve
additional warm-up calls are separate from the 108 conformance forwards.

Annotation evaluation uses 36 source-exact generator-oracle documents. Each has
three identity nodes; each omission is tested separately (108 controls). Another
36 tests combine endorsements of different payloads. Two randomly generated
machine credentials are used, not two human reviewers. An unregistered alias
is retained as a deliberately unsolved negative control.

## 8. Results

All 108 conformance forwards preserved exact matched-name/resegmentation scores
and public activations. All six native-evidence reference comparisons matched.
The scorer remained nonconstant across the three evidence bodies; correctness
of those scores is neither required nor claimed.

| Measurement | Result |
|---|---:|
| Scoring-transformer ratio to native evidence BPE | 1.000x |
| Reference body 1: scalar -> canonical tokens | 281 -> 62 |
| Reference body 2: scalar -> canonical tokens | 291 -> 63 |
| Reference body 3: scalar -> canonical tokens | 284 -> 64 |
| Generated-corpus scalar -> canonical total | 25,201 -> 8,457 |
| Aggregate corpus transformer-token reduction | 66.44% |
| Matched-name/segmentation score gap | 0 |
| Direct counterfactual array comparisons | 3,360, all equal |
| Native evidence hidden and full-vocabulary checks | 6, all equal |
| Omitted registered references blocked | 108/108 |
| Conflicting reviewed payload pairs blocked | 36/36 |
| CPU tests, including 2,000 Unicode property cases | 52 pass |

Across generated cases, old-scalar/native evidence ratios range from 1.39x to
6.37x, with mean per-case ratio 3.57x. The aggregate reduction above is computed
from total token counts, not a mean of these ratios. The three John Miller
reference raw-plus-canonical embedding counts are 133/135/137, versus old scalar
raw embeddings 326/336/329. Extra source embedding work is not hidden by the
native transformer count.

Median measured service time was 334.4 ms scalar versus 81.5 ms canonical.
The median paired speedup was 4.10x, with interquartile range 4.01x-4.37x.
These are local warm timings on one Apple Silicon runtime; no independent
cross-device speedup or population confidence interval is asserted. The final
factory smoke test passed the dual-review public API, process-wide lock,
poisoned-service reuse rejection and final parameter/file integrity checks.

## 9. Limitations and negative controls

The strongest unresolved issue is correct semantic role annotation. In the
synthetic study it is exact by construction. In arbitrary real text a name can
denote an applicant, a client, a tool, an organization or an ambiguous referent.
Shared mistaken endorsements do not remove that ambiguity. The explicit
negative control preserves the alias Leaf when only John was registered; the
program does not claim that this omission was automatically discovered.

Canonicalization is supporting code inside the model backend. It has the same
role-based information-exclusion purpose as anonymization, while retaining
the original named source embedding before the exclusion. It is not an
autonomous model that has learned universal semantic name neutrality. Raw
embeddings and stored name knowledge remain, and actual race is not identified
or erased. Other CV proxies, population correlations and job-policy bias remain.

The proof compares accepted requests under a benign trusted runtime. Error,
admission, timing, resource use and private audit observations are not protected.
Canonical overflow occurs inside the backend and poisons that service; rebuild
before reuse. The exact pinned checkpoint intentionally rejects other exports.
Native-evidence computational parity includes native mistakes and does not
establish useful hiring accuracy or native named-CV score preservation.

The public corpus and all role labels are generated, not independently annotated
real applicants. The evidence bodies are few and repeated. No model training,
independent replication, externally preregistered human-review study or new
theory claim is supplied. The original project used public résumé/job datasets,
but those records are not used or redistributed in this extension.

## 10. Reproducibility and publication status

The release includes a source-hashed plan, actual token/timing records, full
synthetic score metadata, annotation controls, tests, symbolic specifications,
read-only verifiers, the standalone PDF and Markdown source, an artifact manifest
and CPU CI. Large activation arrays are compared during GPU execution but are
not all retained; the CPU verifier replays records/coverage, not independent GPU
inference. The tested model is a separate unadapted MLX affine-four-bit derivative
of Qwen2.5-1.5B-Instruct, with exact parameter digest and file hashes in the package.
Original base files and loaded parameters remain unchanged. Clef and other
training jobs are not used or interrupted. Keys are random in memory and not saved.

This is a portfolio-ready research artifact and preprint draft, not an accepted
or peer-reviewed publication, a DOI-bearing submission, a production hiring
system or proof of academic novelty. Author review, independent replication,
real annotation-quality studies and a venue-appropriate submission are still
needed. Code, experiments and prose were developed with AI coding assistance;
this is disclosed for review. The authored code/documentation are MIT-licensed;
external model, tokenizer, dependency and paper terms remain with their owners.

## References

1. Rico Sennrich, Barry Haddow and Alexandra Birch. Neural Machine Translation
   of Rare Words with Subword Units. ACL 2016.
   https://aclanthology.org/P16-1162/
2. Hugging Face. The Tokenization Pipeline, Tokenizers v0.20.3 documentation.
   https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline
3. Matt J. Kusner, Joshua R. Loftus, Chris Russell and Ricardo Silva.
   Counterfactual Fairness. 2017. https://arxiv.org/abs/1703.06856
4. Stephen Chong, K. Vikram and Andrew C. Myers. SIF: Enforcing Confidentiality
   and Integrity in Web Applications. USENIX Security 2007.
   https://www.cs.cornell.edu/andru/papers/usenix07-html/paper.html
5. Ron Artstein and Massimo Poesio. Inter-Coder Agreement for Computational
   Linguistics. Computational Linguistics 34(4), 2008, pp. 555-596.
   https://aclanthology.org/J08-4004/
