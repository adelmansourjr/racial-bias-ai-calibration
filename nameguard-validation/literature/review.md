> Historical background from the synthetic release, followed by the expanded
> closest-work assessment. Current real-excerpt results and updated baseline
> conclusions are in paper/manuscript.md.

# Literature review: what name-channel isolation establishes

## Review method and boundaries

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

## 1. Names as an experimental intervention in hiring

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

## 2. Fairness is broader than an identity channel

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

## 3. Behavioral equality does not establish representation erasure

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

## 4. Erasure, adapters and structural routing are different interventions

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

## 5. The closest conceptual foundation is information-flow security

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

## 6. Tokenization efficiency and annotation reliability

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

## 7. Evaluation, reporting and research gaps

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

## Comparison of guarantees (our synthesis)

| Approach | What its result concerns |
|---|---|
| Name-swap hiring audits [6-8] | Measured outcomes on specific prompts, names and tasks; no universal erasure theorem |
| INLP / LEACE [13-14] | Specified linear representational criteria; not all decoders or every unseen distribution |
| Quantized LoRA / ReFT [17-18,20] | Efficient adaptation/intervention; fairness requires additional objectives and evaluation |
| Concept bottlenecks [19] | Predictions through supervised concepts; concept correctness remains an assumption |
| Information-flow enforcement [4,12] | Dependence under a policy and trusted boundary; the policy can be socially inadequate |
| This artifact | Equal scoring-path tensors for equal reviewed Q; no automatic semantic discovery or whole-model erasure |


## Expanded closest-work assessment

# Novelty assessment, 3 November 2026

This is a documented assessment, not a claim that a literature search can prove
the nonexistence of earlier work. Primary sources were opened and their relevant
mechanisms compared. We exclude unsupported priority language and distinguish
published experimental claims from our own implementation.

## Closest work missing from the initial review

Karvonen and Marks (2025) [26] directly study hiring fairness through internal
activation interventions. Section 3.4 computes demographic directions and uses
affine concept editing at inference; Sections 3.2-3.3 test selective/contextual
settings and inferred college-affiliation cues. This is closer than generic
LoRA work. Its empirical bias reductions are not an exact universal exclusion
theorem, and our routing architecture is not a reproduction of its ACE results.

Russinovich (2026) [27], Section III-A, already describes a serving-time span veto
with bit-identical outside activations under same-length marked-span substitution.
It explicitly addresses geometry, canonical token boundaries and deterministic
kernels. Thus “the first internal span noninterference mechanism” would be
unsupported. Our variable-name-length canonical evidence route differs in its
token/position convention, but ordinary evidence canonicalization already gives
that functional property. This difference alone is not demonstrated novelty.

Debenedetti et al. (2025) [28] enforce capabilities and information flows around
LLM agents. This supports the broader separation between model behavior and
enforced policy. It is not a hiring study and does not validate our roles.
Tamkin et al. (2023) [29] supply broad demographic decision audits and prompting
mitigations; Marshall et al. (2024; revised 2025) [30] derive ACE for activation
steering. Hoffstedde et al. (2026) [31] evaluate counterfactual names and practical
mitigations in Japanese hiring. These antecedents prevent claiming that
controlled-name audits, activation editing or anonymization are new inventions.

## Claim disposition

| Proposed claim | Assessment |
|---|---|
| First to reduce hiring bias through activations | Reject: direct prior work [26]. |
| First exact internal noninterference for marked text | Reject: close span-veto prior work [27], plus classical information-flow foundations [4,12]. |
| QLoRA guarantees every name is ignored | Reject: no such guarantee follows from adapter efficiency [17-18]; this method trains no adapter. |
| Native-token canonicalization is new theory | Reject: deterministic factorization and standard tokenization explain it. |
| Internal placement is fairer than equivalent external canonicalization | Reject: identical Q and scoring computation imply identical outputs. |
| Reproducible typed-name scoring artifact with source-integrity checks and explicit negative controls | Supported description of this implementation; originality of the integration is not a field-wide priority claim. |
| New controlled real-excerpt conformance/equivalence dataset and measurements | A new local experiment in this release, with sampling and privacy limits; not an independent replication by another lab. |
| Publishable research contribution | Possible artifact/replication/negative-result framing; acceptance, novelty and significance require external expert assessment. |

## Defensible positioning

Use: **“A reproducible audit of name-channel noninterference in a frozen language
model, including a strong equivalent baseline, token-cost accounting, real-excerpt
conformance and a separation between public invariance and private identity
retention.”** Do not present a new universal name-unlearning algorithm.

The mathematical joint-editing caveat in docs/FORMAL-CLAIMS.md follows from
standard constrained least squares. It is a useful audit of interpretation,
not a newly discovered erasure method or a claim that [26]'s results are invalid.

## Search record and remaining scope

Queries included “name noninterference language model fairness resume,”
“counterfactual anonymization resume language model,” “Robustly Improving LLM
Fairness in Realistic Settings via Interpretability,” “span veto language model,”
“bit-identical noninterference models,” and “affine concept editing Marshall.”
Selection favored primary manuscripts with directly comparable mechanisms;
secondary search hits were used only to locate those manuscripts. The earlier
25-source review and its version caveats remain available in the prior release.
This expanded search found material omissions in that review; it does not
establish exhaustive coverage, originality or peer-review acceptance.
