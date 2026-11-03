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
