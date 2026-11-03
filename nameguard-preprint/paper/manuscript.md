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

## 2. Literature review and positioning

### Review method and boundaries

This is a structured narrative review, not an exhaustive systematic review or a
PRISMA study. Sources were checked on 3 November 2026 through publisher pages,
author manuscripts and official documentation. The 25 selected references cover
six questions: why audit names, which fairness property is intended, whether
behavior reveals internal information, what architectural exclusion guarantees,
how to preserve token efficiency, and how to evaluate a defensible artifact.
`search-log.json` records the search families, selection rules and exclusions.
`references.json` records source locations, version caveats and how extensively
each source was inspected. Abstract-only checks are identified rather than
presented as complete paper replications. The bibliography is in `references.bib`.

The review distinguishes a prior paper's findings from our deductions. A citation
motivates a design choice; it does not validate this implementation. We do not
infer the absence of other work from this targeted search. No numerical outcomes
from different models, tasks or populations are pooled into a meta-analysis.

### 2.1 Names as an experimental intervention in hiring

Bertrand and Mullainathan (2004) [6] establish a correspondence-audit precedent:
vary names associated with social groups while controlling resume content and
measure employer responses. This motivates matched-name interventions, not
assigning a person's actual race from a name. Our synthetic model-score checks
are not field evidence about employer callbacks.

Wilson and Caliskan (2024; corrected manuscript 2026) [7] audit resume retrieval
using text-embedding similarity. Their first-page erratum says gender-only
preferences were reversed by a coding error, while race-only and intersectional
results were unaffected. We therefore avoid the uncorrected abstract's gender
figures. Their retrieval setting differs from our autoregressive model readout;
its results motivate auditing but cannot be imported as our baseline bias rate.

Armstrong et al. (2024) [8] study GPT resume assessment and generation, showing why
evaluation should distinguish scoring existing evidence from producing new
candidate descriptions. We study a deterministic restricted scoring interface,
not generated recommendations or an end-to-end recruitment workflow. Together,
these audits justify controlled identity swaps while leaving deployment-wide
claims dependent on task, prompt, model and population.

### 2.2 Fairness is broader than an identity channel

Kusner et al. (2017) [3] formulate counterfactual fairness using a causal model.
Our equivalence relation instead fixes retained evidence Q and varies designated
identity text. We neither identify the causal effect of race nor hold constant
every consequence of discrimination. A job history or address can remain a
racial proxy even when applicant-name substitution cannot affect the score.

Dwork et al. (2012) [11] locate individual fairness in a task-relevant similarity
metric. Our exact equality of evidence is a deliberately narrower relation; it
does not solve how two different histories ought to be compared. Raghavan et al.
(2020) [9] examine hiring vendors' validation claims, making job relevance and
validation choices part of the research question rather than assumed outputs.
Selbst et al. (2019) [10] warn against abstractions that exclude the surrounding
social process. This means reviewer authority, applicant appeal, retained proxies
and selection policy belong in our limitations, not outside the system boundary.

The defensible conclusion is conditional name-channel noninterference, not
"truly unbiased hiring." Equal scores can be equally inaccurate. A harmful job
rubric stays harmful when its name input is disconnected. Real deployment needs
validity evidence, stakeholder review, accessible correction and subgroup audits
in addition to this narrow technical property.

### 2.3 Behavioral equality does not establish representation erasure

Hewitt and Liang (2019) [15] show why probe controls and selectivity matter: probe
success can partly reflect the probe's learning ability. Conversely, failure of
one fitted probe does not certify that no decoder can recover a concept. Gonen
and Goldberg (2019) [25] demonstrate residual gender structure after particular
word-embedding debiasing procedures. That finding concerns those embeddings;
we use it as a warning against generalizing one favorable bias metric.

Our simplest counterexample is H(q,n)=(q,n), with output S(H)=q. Swapping n cannot
change S, yet n is perfectly recoverable from H. Thus removing observable score
bias does not imply removing activation-space information. Even information
about names is not itself an unfair decision: knowing a name's linguistic origin
is compatible with prohibiting its influence on a hiring score. The distinction
is encoding versus causal use, under a stated intervention.

Amnesic probing (Elazar et al., 2021) [16] links representation interventions to
behavioral changes, rather than treating a probe alone as an explanation. Our
public-stage equality measurements similarly examine more than a final score,
but only after the explicit evidence boundary. Raw source embeddings, model
weights and any private diagnostic branch remain outside that erasure claim.

### 2.4 Erasure, adapters and structural routing are different interventions

INLP (Ravfogel et al., 2020) [13] repeatedly fits linear attribute predictors and
projects representations into their nullspaces. LEACE (Belrose et al., 2023;
manuscript v4, 2025) [14] derives a closed-form minimal-distortion linear eraser
under a precise guardedness definition. Its convex-loss linear-prediction
guarantee must not be restated as unrestricted independence or inability of
every zero-one-loss threshold classifier to recover a label.

For example, balanced Z with X=-1 or +1 when Z=0, and X=0 when Z=1 has equal
class means and zero covariance. Nevertheless, thresholding X>0 predicts Z=0
with 75% accuracy; a nonlinear decoder of |X| recovers Z exactly. This is our
finite mathematical witness, not a replication or refutation of LEACE's stated
theorem. It illustrates why the predictor family and loss must accompany the
word "erasure." Sample-estimated erasers also require held-out leakage tests.

LoRA (Hu et al., 2022) [17] freezes base weights and learns low-rank updates.
QLoRA (Dettmers et al., 2023) [18] combines frozen quantized weights with adapters
and specific memory-saving techniques. Our earlier MLX affine-four-bit adapter
experiments should be called quantized LoRA, not a faithful NF4/double-quantized
QLoRA replication. Neither adapter efficiency method by itself provides a
universal fairness guarantee. This final canonical-evidence release trains no
adapter and does not change base parameters.

Concept bottleneck models (Koh et al., 2020) [19] and ReFT (Wu et al., 2024) [20]
offer alternative controlled representations or interventions. They help frame
future learned evidence extractors, but are not evaluated baselines here.
The present tradeoff is explicit: deterministic isolation gives a conditional
dependence guarantee while shifting semantic correctness to reviewed roles.

### 2.5 The closest conceptual foundation is information-flow security

Sabelfeld and Myers (2003) [12] explain end-to-end information flow; Chong et al.
(2007) [4] distinguish confidentiality and integrity in an enforced system.
These are closer foundations for our theorem than statistical debiasing.
Let D~D' mean Q(D)=Q(D'). A score S=G(Q(D)) is constant on each equivalence class.
The proof is ordinary deterministic composition, not a new fairness theorem.
Authenticated reviews protect integrity of the chosen partition; they cannot
prove that the partition faithfully captures applicant identity.

This framing yields an essential baseline. An external canonicalizer computing
the identical Q followed by the same tokenizer, weights, positions and readout
has the same score as our internal canonicalizer. Embedding the raw source first
does not strengthen that output guarantee. It is a trace/design constraint and
adds work; names still exist in the raw private activations. Position, timing,
admission, error and audit channels need separate treatment. A correct public
score theorem is not a claim that the entire computer leaks no identity.

Internal placement is therefore an implementation characteristic, not evidence
of novelty or model unlearning. The artifact's contribution is an inspectable,
tested integration of reviewed source roles, native evidence tokenization and
frozen-model scoring, with explicit failure cases and reproducible records.

### 2.6 Tokenization efficiency and annotation reliability

Sennrich et al. (2016) [1] motivate subword segmentation for open-vocabulary NLP;
the official Tokenizers pipeline [2] separates transformations that can affect
boundaries. Our deduction is algebraic: a codec satisfying C(a+b)=C(a)||C(b) for
every scalar split is forced to encode scalars independently. It generally cannot
also preserve every merge of whole-string BPE. Joining reviewed evidence before
one tokenizer call avoids this incompatible requirement.

The exact result is n(Q)=|T(Q)| transformer positions, not globally minimum tokens
and not total compute parity with unrestricted input. Raw-source embedding and
review verification remain overhead. The measured 4.10x median paired speedup
compares our two service packages; it is not a speedup over a strong optimized
external-canonicalization baseline, which was not timed. It is also not an
isolated ablation of token count because review APIs differ.

Artstein and Poesio (2008) [5] distinguish agreement measures and interpretation.
Two credentials agreeing can share one error. Our synthetic oracle checks span
mechanics, registered omissions and conflicting payloads, not human semantic
accuracy. Independent adjudication must measure whole-document identity
omissions and false evidence removals, including unknown aliases, transliteration,
same-spelled skills and ambiguous third parties. Character agreement alone can
look excellent while missing a short identifying span.

### 2.7 Evaluation, reporting and research gaps

CheckList (Ribeiro et al., 2020) [21] motivates targeted behavioral tests alongside
aggregate accuracy. We separate name invariance, evidence sensitivity, annotation
failure controls, public-stage equality, native-evidence parity and efficiency.
The 108 controlled forwards repeat just three evidence bodies; their 3,360 array
comparisons are correlated engineering checks, not independent fairness trials.
Zero observed failures do not yield a population failure-rate estimate without
a sampling model. The universal conditional argument comes from architecture,
not extrapolation from those tests.

Dodge et al. (2019) [22] motivate reporting development choices and computation
budgets. We preserve the source freeze and distinguish inherited GPU measurements
from this later literature audit. Model cards (Mitchell et al., 2019) [23] and
datasheets (Gebru et al., 2021) [24] motivate explicit intended use, limitations
and data provenance. The final conformance corpus is generated; the supplied
public CV/job datasets belonged to earlier project exploration and do not supply
real-world validation for this extension.

Before an empirical fairness publication, the main missing evidence is an
independent real-document role study, a useful hiring-label benchmark and a
competitive external-canonicalization baseline. Before a systems publication,
the main missing evidence is a sharper threat model, implementation-level
verification or independent audit, and broader workloads/hardware. The current
artifact supports a transparent technical-report/preprint draft; novelty and
venue suitability remain reviewer judgments.

### Comparison of guarantees (our synthesis)

| Approach | What its result concerns |
|---|---|
| Name-swap hiring audits [6-8] | Measured outcomes on specific prompts, names and tasks; no universal erasure theorem |
| INLP / LEACE [13-14] | Specified linear representational criteria; not all decoders or every unseen distribution |
| Quantized LoRA / ReFT [17-18,20] | Efficient adaptation/intervention; fairness requires additional objectives and evaluation |
| Concept bottlenecks [19] | Predictions through supervised concepts; concept correctness remains an assumption |
| Information-flow enforcement [4,12] | Dependence under a policy and trusted boundary; the policy can be socially inadequate |
| This artifact | Equal scoring-path tensors for equal reviewed Q; no automatic semantic discovery or whole-model erasure |

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

### 5.1 Boundary equivalence and what is not erased

Let G be the identical evidence tokenizer, embedding, transformer and readout.
An external reviewed canonicalizer followed by G computes G(Q(D)); the internal
canonicalizer computes that same function. Their scores are therefore equal
when implementation conventions match. Embedding the full named source first
adds no mathematical score-isolation advantage. We did not time a strong
external-Q baseline, so the measured speedup must not be presented as superiority
over optimized anonymization. Internal placement is an implementation boundary,
not a demonstration of autonomous learned name removal.

Behavioral neutrality does not imply internal erasure: H(q,n)=(q,n), S(H)=q
keeps the name perfectly recoverable while producing name-invariant scores.
The raw branch here similarly retains name information. Public scoring tensors
are equal under the stated intervention; the original model's knowledge is not
erased. A second exact witness has balanced classes Z, X in {-1,+1} for Z=0
and X=0 for Z=1. Class means coincide and covariance is zero, yet a threshold
achieves 3/4 accuracy and a nonlinear decoder recovers Z perfectly. This does
not contradict LEACE's specified convex-loss definition [14]. Five executable
finite witnesses in scripts/scientific_claims.py protect these distinctions;
they are mathematical examples, not extra model experiments.

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

### 10.1 Remaining publication requirements

This revision adds a checked literature synthesis, five exact finite witnesses
and six citation checks (63 CPU tests in total), preserving the
original model experiment and source hashes. Its 25-source review establishes
context, not exhaustive coverage or priority. The closest baseline is equivalent
external canonicalization; the artifact does not establish a novel unlearning
algorithm. Independent human role annotation, a useful hiring-label benchmark,
matched external-Q timings and independent replication remain necessary for
stronger empirical claims. A clustered statistical confirmation plan and a
claim-by-claim evidence ledger are supplied in docs/CLAIM-EVIDENCE.md. A completed
local source freeze must not be described as external preregistration.

## References

1. Rico Sennrich, Barry Haddow, Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. ACL. DOI: 10.18653/v1/P16-1162. https://aclanthology.org/P16-1162/ Accessed 2026-11-03.

2. Hugging Face. n.d.. The Tokenization Pipeline. Tokenizers documentation, version 0.20.3 (version pinned; publication date unspecified). https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline Accessed 2026-11-03.

3. Matt J. Kusner, Joshua R. Loftus, Chris Russell, Ricardo Silva. 2017. Counterfactual Fairness. NeurIPS. https://arxiv.org/abs/1703.06856 Accessed 2026-11-03.

4. Stephen Chong, K. Vikram, Andrew C. Myers. 2007. SIF: Enforcing Confidentiality and Integrity in Web Applications. USENIX Security. https://www.cs.cornell.edu/andru/papers/usenix07-html/paper.html Accessed 2026-11-03.

5. Ron Artstein, Massimo Poesio. 2008. Inter-Coder Agreement for Computational Linguistics. Computational Linguistics 34(4), 555-596. DOI: 10.1162/coli.07-034-R2. https://aclanthology.org/J08-4004/ Accessed 2026-11-03.

6. Marianne Bertrand, Sendhil Mullainathan. 2004. Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination. American Economic Review 94(4), 991-1013; NBER working paper 9873 (2003). DOI: 10.1257/0002828042002561. https://www.nber.org/papers/w9873 Accessed 2026-11-03.

7. Kyra Wilson, Aylin Caliskan. 2024. Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval. AIES 7(1); corrected author manuscript v3, 30 August 2026. DOI: 10.1609/aies.v7i1.31748. https://arxiv.org/pdf/2407.20371v3 Accessed 2026-11-03.

8. Lena Armstrong, Abbey Liu, Stephen MacNeil, Danaë Metaxa. 2024. The Silicon Ceiling: Auditing GPT's Race and Gender Biases in Hiring. ACM; author manuscript v3, 15 November 2024. DOI: 10.1145/3689904.3694699. https://arxiv.org/abs/2405.04412v3 Accessed 2026-11-03.

9. Manish Raghavan, Solon Barocas, Jon Kleinberg, Karen Levy. 2020. Mitigating Bias in Algorithmic Hiring: Evaluating Claims and Practices. FAT*; author manuscript arXiv:1906.09208v3 (2019). https://arxiv.org/abs/1906.09208v3 Accessed 2026-11-03.

10. Andrew D. Selbst, danah boyd, Sorelle A. Friedler, Suresh Venkatasubramanian, Janet Vertesi. 2019. Fairness and Abstraction in Sociotechnical Systems. FAT*. DOI: 10.1145/3287560.3287598. https://friedler.net/papers/sts_fat2019.pdf Accessed 2026-11-03.

11. Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Richard Zemel. 2012. Fairness Through Awareness. ITCS; author manuscript arXiv:1104.3913 (2011). https://arxiv.org/abs/1104.3913 Accessed 2026-11-03.

12. Andrei Sabelfeld, Andrew C. Myers. 2003. Language-Based Information-Flow Security. IEEE Journal on Selected Areas in Communications 21(1). https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf Accessed 2026-11-03.

13. Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg. 2020. Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. ACL. DOI: 10.18653/v1/2020.acl-main.647. https://aclanthology.org/2020.acl-main.647/ Accessed 2026-11-03.

14. Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman. 2023. LEACE: Perfect linear concept erasure in closed form. NeurIPS; author manuscript v4, 3 April 2025. https://arxiv.org/pdf/2306.03819v4 Accessed 2026-11-03.

15. John Hewitt, Percy Liang. 2019. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP. DOI: 10.18653/v1/D19-1275. https://aclanthology.org/D19-1275/ Accessed 2026-11-03.

16. Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. 2021. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. Transactions of the Association for Computational Linguistics. DOI: 10.1162/tacl_a_00359. https://aclanthology.org/2021.tacl-1.10/ Accessed 2026-11-03.

17. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR; author manuscript arXiv:2106.09685v2 (2021). https://arxiv.org/abs/2106.09685v2 Accessed 2026-11-03.

18. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. https://arxiv.org/abs/2305.14314 Accessed 2026-11-03.

19. Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang. 2020. Concept Bottleneck Models. ICML. https://arxiv.org/abs/2007.04612 Accessed 2026-11-03.

20. Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts. 2024. ReFT: Representation Finetuning for Language Models. Author manuscript arXiv:2404.03592. https://arxiv.org/abs/2404.03592 Accessed 2026-11-03.

21. Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL. https://arxiv.org/abs/2005.04118 Accessed 2026-11-03.

22. Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith. 2019. Show Your Work: Improved Reporting of Experimental Results. EMNLP-IJCNLP. https://arxiv.org/abs/1909.03004 Accessed 2026-11-03.

23. Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru. 2019. Model Cards for Model Reporting. FAT*; author manuscript arXiv:1810.03993. https://arxiv.org/abs/1810.03993 Accessed 2026-11-03.

24. Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, Kate Crawford. 2021. Datasheets for Datasets. Communications of the ACM; author manuscript arXiv:1803.09010. https://arxiv.org/abs/1803.09010 Accessed 2026-11-03.

25. Hila Gonen, Yoav Goldberg. 2019. Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them. NAACL-HLT. DOI: 10.18653/v1/N19-1061. https://aclanthology.org/N19-1061/ Accessed 2026-11-03.
