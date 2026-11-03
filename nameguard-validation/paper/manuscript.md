# A Reproducible Audit of Name-Channel Noninterference in a Frozen Language Model

Adel Mansour | Controlled real-excerpt validation and preprint draft | 3 November 2026

## Abstract

We audit structural name-channel isolation in a frozen Qwen2.5-1.5B derivative,
extending an earlier generated-fixture study with 32 real CV excerpts, eight
occupations, two job contexts per CV and four controlled header names. A frozen
pre-inference plan specifies a 768-forward experiment, direct public-state
comparisons and an equivalent external-canonicalization baseline. Original
named inputs show a mean within-CV probability range of 9.75 percentage points;
name swaps flip binary decisions in 9/64 contexts. The role-isolated scoring path
has zero matched-name score variation; all 6,144 direct counterfactual public-stage
comparisons and 128 native-evidence hidden/vocabulary comparisons match exactly.
A category-pair relevance diagnostic is 62.5% for named scoring and 68.75% after
isolation, but its paired bootstrap interval includes no improvement; it is not
human hiring accuracy. Strong-baseline timing is approximately equal, with a
median paired internal/external ratio of 1.002. Raw private embeddings still distinguish
all four names, proving that decision-path invariance is not whole-model name
erasure. An expanded 31-source review identifies close prior work and limits the
contribution to a reproducible implementation and audit. No universal racial
fairness, independent human validation or first-method claim is established.

## 1. Research question and contribution

Can a frozen language-model scorer be made exactly invariant to a designated
applicant-name channel, while retaining its native computation on the same
evidence? We distinguish this architectural question from two harder questions:
whether every identity reference in arbitrary prose can be discovered, and
whether the resulting hiring policy is substantively fair. The latter questions
require evidence not supplied by name swaps alone.

This report extends an earlier synthetic implementation study with real CV and
job excerpts, a strong equivalent external-canonicalization baseline, clustered
analysis, and a wider novelty assessment. Its contribution is an auditable
experiment and implementation artifact. It is not a claim to invent activation
editing, anonymization or noninterference, and it does not establish whole-model
racial-bias erasure. The author's single-runtime validation is distinct from
independent human evaluation or replication by another research group.

## 2. Literature review and novelty assessment

### Review method and boundaries

The following background synthesis distinguishes the prior synthetic release
from the new real-excerpt experiment. This is a structured narrative review, not an exhaustive systematic review or a
PRISMA study. Sources were checked on 3 November 2026 through publisher pages,
author manuscripts and official documentation. The original 25 selected references cover
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
review verification remain overhead. The earlier synthetic release's measured 4.10x median paired speedup
compares our two service packages; it is not a speedup over a strong optimized
external-canonicalization baseline. That baseline was not timed in the earlier
release; the new experiment below now measures it. It is also not an
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
The prior release's 108 controlled forwards repeat just three evidence bodies;
its 3,360 array
comparisons are correlated engineering checks, not independent fairness trials.
Zero observed failures do not yield a population failure-rate estimate without
a sampling model. The universal conditional argument comes from architecture,
not extrapolation from those tests.

Dodge et al. (2019) [22] motivate reporting development choices and computation
budgets. We preserve the source freeze and distinguish inherited GPU measurements
from this later literature audit. Model cards (Mitchell et al., 2019) [23] and
datasheets (Gebru et al., 2021) [24] motivate explicit intended use, limitations
and data provenance. The earlier release's conformance corpus was generated; the supplied
public CV/job datasets belonged to earlier project exploration and did not supply
validation for that earlier extension. The new controlled study below now uses
real excerpts without claiming human-validated hiring usefulness.

The prior review identified an independent real-document role study, a useful
hiring-label benchmark and a competitive external-canonicalization baseline as
missing evidence. The present experiment supplies controlled real-excerpt
conformance and the strong external baseline; the human-label studies remain
unfulfilled. Before a systems publication,
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


### Closest newer work and the revised novelty conclusion

Karvonen and Marks (2025) [26] directly mitigate hiring bias through affine
activation edits, including more realistic contextual and proxy settings.
Russinovich (2026) [27] already supplies a marked-span veto with exact outside
activation noninterference under fixed geometry. Its canonical boundaries and
length assumptions are explicit. These are material omissions from the initial
review: we now reject broad first-method claims. Canonical evidence routing
handles variable raw-name lengths by rebuilding its scoring sequence, but that
functional property also follows from ordinary external canonicalization.

CaMeL (Debenedetti et al., 2025) [28] places capability-based flow enforcement
around LLM agents. Tamkin et al. (2023) [29] audit demographic effects across
decision scenarios; Marshall et al. (2024; revised 2025) [30] derive affine
concept editing for activation steering; Hoffstedde et al. (2026) [31] test
counterfactual names and mitigation in Japanese hiring. These sources support
a narrow artifact/audit contribution, not a new unlearning algorithm. The
companion novelty matrix records rejected priority claims and search queries.

## 3. Architecture and formal target

Let D contain exact source spans tagged as identity or retained evidence. Let
X(D) concatenate all source text and Q(D) concatenate only retained spans. With
the original tokenizer T, embeddings E and fixed scoring network G, execution is:

    R(D) = E(T(X(D)))        # complete named source embedded first
    H0(D) = E(T(Q(D)))       # evidence rebuilt inside the backend
    H_l(D) = F_l(H_(l-1)(D)) # original 28 transformer blocks
    S(D) = original_readout(H_L(D)).

R supplies no tensor, position, mask, cache or length to the evidence branch.
Names enter original raw embeddings intact. Explicit code inside the backend
constructs Q from reviewed roles; the model has not learned to discover every
name autonomously. No adapter is trained in this release. Base parameters and
checkpoint files are verified before and after inference.

For two accepted documents with Q(D)=Q(D'), deterministic tokenization gives
identical evidence IDs. Equal embeddings, masks and positions propagate through
each unchanged layer, giving equal scoring states and scores. The statement
covers admitted unseen name values under the role/evidence premise. It does not
cover undeclared body aliases, errors, admission, timing or private diagnostics.

An external canonicalizer that produces the same Q followed by the same G has
the same output. Raw-source embedding first adds no mathematical score-isolation
advantage. It is an implementation constraint with additional work, not a learned
erasure mechanism. The transformer length is exactly |T(Q)|; total embedding
visits include both |T(X)| and |T(Q)|. No globally minimum-token claim follows.

## 4. Frozen controlled study on real excerpts

The study plan and 12 inference/preparation source hashes were frozen before
model inference. We sampled four CV records in each of eight categories from
the provided Kaggle Resume Dataset, using deterministic hash ordering with seed
20261103. The source contains 2,484 records. We excluded the 24 original datathon
case IDs, but do not certify disjointness from every prior local study or model
pretraining. No fitting, adapter selection or outcome-dependent case selection
was performed.

Each CV contributes two contexts: a category-matched job and a deterministically
mismatched category's job, drawn from the cached 61-row LinkedIn listings sample.
Categories are accountant, advocate, banking, engineering, healthcare, IT, sales
and teacher. Two CVs per category use a plain instruction and two a selective
instruction. Since styles use different CVs, their comparison is descriptive,
not a causal within-CV estimate of selective-context effects.

Inputs contain the first 140 whitespace words of real CV text and the first 40
words of a real job description, plus job title and a fixed A/B relevance query
in the original chat template. These are explicitly excerpt tasks, not full-CV
validation. The four inserted header names are John Miller, Mohammed Hassan,
Wei Chen and Neruvo Qaslin. The last is a constructed name, not certified absent
from pretraining. Names are lexical interventions, not verified racial labels.

The inserted name span is exact by construction and authenticated using two
temporary machine credentials. These are not human annotators. Original body
names and other proxies are retained unchanged. The experiment therefore tests
the declared header channel with real evidence, not automatic name discovery
throughout natural CVs. CV/job text and keys remain local and are excluded from
the public artifact; source hashes and selection code permit data-authorized
reproduction.

For every context, we run the corrected path and original named-input scorer
under all four names: 256 rows, 512 forwards. Each context also receives a native
external-Q reference forward (64), comparing the entire final hidden array and
full final-position vocabulary output. Corrected counterfactuals compare token
IDs, embeddings, all 28 blocks, final normalization and full vocabulary arrays
directly. There are 64 reference variants and 192 counterfactuals: 6,144 direct
public-stage comparisons, not 6,144 independent applicants.

Warm timing uses one paired trial per CV on its matched job, after two warmups
per method, with shuffled method order. Both methods verify the same signed role
payload and model structure. Internal scoring additionally prepares and embeds
the original source. The external baseline builds Q outside the backend and uses
the native model. Neither timed path copies observed activation arrays. Signing
and model loading are excluded. The run is sequential, not a concurrency or
load-throughput benchmark. Timing adds 64 measured and 128 warmup forwards, for
768 actual forwards in total.

## 5. Results

All frozen-plan contexts completed without outcome-dependent exclusions. The
checkpoint's original parameter digest and file hashes remained unchanged.

| Measurement | Observed result |
|---|---|
| Real CV clusters / job contexts / name rows | 32 / 64 / 256 |
| Actual forwards including references, timing and warmups | 768 |
| Maximum corrected probability range across names | 0, exact |
| Raw mean CV-level probability range | 9.75 percentage points |
| Bootstrap interval for that mean | 8.27 to 11.19 percentage points |
| Largest raw context probability range | 34.33 percentage points |
| Raw versus corrected binary-flip contexts | 9/64 versus 0/64 |
| Direct counterfactual public-stage comparisons | 6,144, all equal |
| Native-evidence final-hidden/full-vocabulary comparisons | 128, all equal |
| Category-pair relevance: raw / corrected | 62.5% / 68.75% |
| Paired relevance difference, bootstrap interval | +6.25 points, interval -0.78 to +14.06 points |
| Internal / external median service latency | 303.5 / 299.6 ms |
| Median paired internal/external ratio | 1.002 |
| Pairwise distinct private raw-name states | 4/4 names in every context |

The raw range measures naming-cue sensitivity, not a measured racial group
outcome disparity. A positive range alone does not identify racism or its
mechanism. The corrected exact equality follows from the controlled evidence
boundary and is checked on actual arrays. It is not extrapolated statistically
from four known names to every unannotated name in arbitrary prose.

The median paired timing ratio has interquartile range
0.994-1.005. The strong external-Q baseline therefore receives
no claim of speed inferiority. The earlier 4.10x speedup against scalar encoding
must not be applied to this comparison. These warm single-device timings are
not a multi-device equivalence test or production throughput result.

## 6. Statistical interpretation and performance

The independent resampling unit is a CV, retaining its two jobs and four names
together. We resample four CVs with replacement within each of the eight fixed
categories, 5,000 times with a fixed seed, and report percentile intervals.
These describe resampling stability in this small stratified dataset study,
not worldwide population fairness. Repeated activation checks are not treated
as independent Bernoulli trials. No universal failure-rate confidence bound is
inferred from observing zero score gaps.

Category-pair relevance asks whether a CV's matched-category job receives a
higher A/B margin than its mismatched job. Ties receive half credit. Each CV's
raw result averages its four name variants. This is a narrow diagnostic based
on source category labels; category matching is not human hiring accuracy or
job-performance validity. Native evidence parity is tested separately through
complete hidden/vocabulary arrays, not inferred from that diagnostic.

The observed paired relevance difference does not establish a population
improvement: its bootstrap interval includes zero. No utility noninferiority
margin was prespecified with domain experts, so we do not claim a formal hiring
utility noninferiority result. We do establish native-evidence computational
parity on the tested contexts. This includes the native model's mistakes.

## 7. Why score isolation is not whole-model erasure

For fixed Q, all four actual raw embedding states are pairwise distinct, while
the scoring states coincide. Under the experiment's uniform four-name
intervention distribution, the map from the name to raw state is injective.
Consequently H(N|H_raw,Q)=0 and I(N;H_raw|Q)=log2(4)=2 bits. The invariant public
state has I(N;H_public|Q)=0 for that same finite distribution. These statements
concern this controlled name variable, not actual race, an open-world decoder
or every private/timing channel. Distinctness uses direct array comparison,
including shape, rather than hashes alone.

If original name embeddings remain distinct, an unchanged raw embedding stage
cannot simultaneously be independent of the name. Later adapters cannot change
the state that already existed. Thus whole-execution name erasure conflicts
with the requirement to preserve that named raw stage. Keeping general name
knowledge is compatible with isolating a hiring score from the identity channel.

A second boundary concerns performance claims. If two native named scores are
a and b, any common corrected score c obeys

    max(|c-a|, |c-b|) >= |a-b|/2.

This follows from the triangle inequality. Exact name invariance and exact
preservation of unequal original scores are incompatible. Preserving task
accuracy is a separate empirical goal; preserving native evidence computation
is the exact target of this architecture.

The companion mathematical note also audits summed affine editing in [26]. For
unit direction columns U and target b, a one-pass summed edit has residual
(I-U^T U)(U^T h-b). It does not generally impose every target when directions
are not orthogonal. Standard constrained least squares replaces that sum with
U(U^T U)^(-1)(U^T h-b) for full-column-rank U. This known algebraic distinction
does not refute [26]'s empirical results and supplies no unrestricted erasure
guarantee. Exact rational examples are included in CPU tests.

## 8. Novelty, validity threats and publication status

The expanded search found direct activation-editing hiring research [26] and a
closely related exact marked-span noninterference mechanism [27] omitted from the
initial review. Accordingly we reject claims to be the first internal fairness
intervention or first exact text-exclusion architecture. The limited contribution
is an inspectable implementation and a controlled real-excerpt audit with a strong
equivalent baseline and explicit negative result about whole-model erasure.
Academic novelty and significance remain external reviewer judgments.

The main threats are small convenience coverage, eight occupations, four fixed
names, repeated contexts, one pinned affine-four-bit Qwen derivative, one device,
an A/B relevance readout, source truncation by declared excerpt policy and
generator-known inserted roles. We do not evaluate free-form name detection,
independent human role labels, real hiring outcomes, calibrated qualification
scores, downstream shortlist fairness or every other proxy. Existing background
names in Q can still influence outputs. Preserving Q under a header swap does
not erase racial associations in qualifications or enforce a fair job policy.

The two credentials ensure mechanical payload agreement and integrity, not two
independent judgments. No study participants, inter-annotator agreement or
external peer-review approval are invented. The independent-review packet states
what a separate evaluator must do. A new research release and polished citations
do not establish acceptance or make this a deployment-ready hiring system.

## 9. Reproducibility and ethical handling

The release contains the pre-inference plan, source hashes, 256 public name rows,
64 native-reference records, paired timing records, clustered analysis, tests,
data provenance, the novelty matrix and read-only verification. It preserves
the previous releases and their failed or limited results. The model parameters
and pinned files passed post-run integrity checks. Clef and other training jobs
were not changed or interrupted. No base weights or private credentials are
distributed. The text and code were developed with AI assistance.

Public data access does not eliminate privacy interests. Raw CV/job excerpts
remain in an ignored local directory; public records contain only source hashes,
categories, model outputs and the four artificial intervention names. Reproduction
requires separately obtaining the source datasets and exact checkpoint. Cached
job sampling limits representativeness and its redistribution terms are not
assumed. Original source terms remain with the providers.

The code is suitable for inspection and reproducible research, not consequential
automated hiring. Stronger claims require independent replication, adjudicated
semantic roles, domain-expert job labels and a new prospective study. The present
report supplies actual controlled real-data measurements and a documented
novelty assessment, rather than treating those tasks as completed by assertion.

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

26. Adam Karvonen, Samuel Marks. 2025. Robustly Improving LLM Fairness in Realistic Settings via Interpretability. Author preprint (version pinned). https://arxiv.org/html/2506.10922v1 Accessed 2026-11-03.

27. Mark Russinovich. 2026. What the Model Cannot See: Covert and Provable Text Invisibility in LLMs. Author preprint (version pinned). https://arxiv.org/html/2610.15883v1 Accessed 2026-11-03.

28. Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr. 2025. Defeating Prompt Injections by Design. Author preprint (version pinned). https://arxiv.org/abs/2503.18813v2 Accessed 2026-11-03.

29. Alex Tamkin, Amanda Askell, Liane Lovitt, Esin Durmus, Nicholas Joseph, Shauna Kravec, Karina Nguyen, Jared Kaplan, Deep Ganguli. 2023. Evaluating and Mitigating Discrimination in Language Model Decisions. Author preprint (version pinned). https://arxiv.org/abs/2312.03689v1 Accessed 2026-11-03.

30. Thomas Marshall, Adam Scherlis, Nora Belrose. 2024. Refusal in LLMs is an Affine Function. Author preprint (version pinned). https://arxiv.org/abs/2411.09003v3 Accessed 2026-11-03.

31. Serena A. Hoffstedde, Machiko Hirota, Akshara Nadayanur Sathis Kanna, Rihito Kotani, Ujwal Kumar, Gabriele Trovato, Phan Xuan Tan. 2026. Gender Bias in LLM Hiring Decisions: Evidence from a Japanese Context and Evaluation of Mitigation Strategies. Author preprint (version pinned). https://arxiv.org/abs/2606.18649v1 Accessed 2026-11-03.
