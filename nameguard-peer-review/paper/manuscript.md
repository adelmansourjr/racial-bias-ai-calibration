# Beyond Equal Scores: Admission-Aware Name Noninterference in Frozen Transformer Scoring

**Technical-report / preprint draft for external review**

Adel Mansour - author details and contribution statement to be confirmed

9 November 2026

## Abstract

Exact equality among accepted applicants can conceal name-dependent rejection.
We study this distinction in a frozen 1.5B-parameter Qwen CV-scoring prototype.
We formulate decision-and-score outcome noninterference over rejection and score, and
implement a finite-panel compiler that chooses a maximum-support evidence-token
contract under an explicit reviewed whitespace policy. The compiler is fitted
using twelve names without model scores, then evaluated on twenty-four disjoint
names across eight layouts. It admits 170/192 test variants, versus 158/192 for a
greedy anchor. In a separate prospectively frozen GPU study, all 192 named inputs
across thirty-two synthetic CV conditions are admitted, with exact scores and
aligned scoring states across six names. An equivalent external retained-token
baseline matches every checked stage. On 100 official ARC-Challenge questions,
the native and protected paths both score 68/100 with identical full-vocabulary
outputs. Credential-task accuracy nevertheless falls from 179/192 to 168/192.
These results support conditional name-channel isolation and expose admission,
encoding and utility limitations. They do not establish automatic name discovery,
whole-model racial-bias erasure or validated real hiring performance.

## 1. Introduction

A hiring model can assign identical scores to different names while rejecting
some naming conventions before scoring. It can also produce equal decisions
while retaining identity information in hidden states. These distinctions matter
when exact fairness claims are attached to a real inference pipeline.

Our question is deliberately narrower than complete racial fairness: **Can a
frozen transformer score be invariant to changes in designated applicant-name
fields, and what limits the coverage of that guarantee when native tokenization
is preserved?** An earlier implementation embedded the original named source,
zeroed designated identity activations and ran native transformer blocks on
ordered evidence rows. This study preserves that implementation and focuses on
its admission boundary and empirical evaluation.

We contribute a score-blind finite-panel evidence-contract compiler, a precise
decision-and-score outcome criterion, a disjoint-name tokenizer audit, and a publicly
reproducible synthetic/GPU comparison against a strong external baseline. The
mathematical ingredients are standard enumeration, information-flow isolation
and projection identities. The proposed originality lies in this explicit
native-token feasibility formulation and audit artifact, not in inventing
anonymization, selective fairness or internal concept editing.

## 2. Related work and positioning

Name substitution can change tokenization as well as demographic associations.
Wolfe and Caliskan [1], An and Rudinger [2], and Nayeem and Rafiei [3] provide
direct antecedents for examining these lexical differences. We therefore report
actual name-string interventions rather than treating script as a racial label.

Karvonen and Marks [4] study hiring fairness using internal concept editing;
Russinovich [5] establishes exact marked-span isolation under explicit runtime
conditions. Classical noninterference supplies the broader foundation [6].
These works rule out claims that we introduced activation-based hiring
mitigation or exact internal isolation. LEACE concerns a specified linear
concept-erasure criterion [7]; QLoRA is an adaptation method [8]. Neither name
alone establishes universal name noninterference.

Selective classification already has a fairness literature: Jones et al. [9]
and Lee et al. [10] demonstrate why coverage and abstention cannot be ignored.
Our restricted compiler addresses the feasibility of a single exact retained-ID
contract under native-BPE boundary constraints. Token gathering also has prior
work [11] and existed in our earlier code. The full contribution map and targeted
search record are in literature/review.md. No field-wide priority is claimed.

## 3. Problem and formal results

### 3.1 Roles, witnesses and complete outcomes

For an original source x, a reviewed applicant-identity interval can absorb at
most sixteen adjacent whitespace characters per side if the review policy
permits it. No substantive evidence character or original source token is
rewritten. A candidate must satisfy native token factorization into evidence,
identity and evidence runs. Its ordered retained native IDs form e.

Let a(x) indicate admission and f(x) be the score when admitted. Define
O(x)=REJECT when a(x)=0 and O(x)=(ACCEPT,f(x)) otherwise. Decision-and-score outcome
invariance holds exactly when admission is constant over the substitution family
and admitted scores are equal. This follows immediately from equality of tagged
outcomes. All-rejection has zero coverage and is reported separately from a
successful scoring result.

For name distributions P,Q, the total variation distance of complete outcomes
is at least the absolute difference in acceptance probability: acceptance is
itself an observable event. This elementary bound explains why conditional score
parity alone cannot exclude a rejection disparity. The defined observer excludes
response timing, token counts, distinct error reasons and other metadata; those
would require a stronger interface theorem.

### 3.2 Maximum-support contract compilation

For panel member i let A_i be the exact retained-ID sequences obtainable under
all permitted native-compatible boundary witnesses. A contract e structurally supports member
i if and only if e belongs to A_i. For E equal to the union of these sets, choose

    e* = argmin_{e in E} (-sum_i 1[e in A_i], K(e), lex(e)),

where K is the sum of minimum witness-formatting costs over supported members.
Enumerating E attains maximum panel support; minimizing each supported member's
cost independently gives the second objective. Duplicate witnesses never count
one member twice. If E is empty, compilation fails closed.

A single contract structurally supports the entire panel if and only if the
intersection of all A_i is nonempty. Otherwise n-max_e support(e) is the exact
structural incompatibility count for this policy and panel. This is not a lower bound for every possible fairness
architecture. These are structural support and rejection bounds. Actual service admission
also requires source-bound semantic approval of a matching witness. Optimizing
actual approvals would require the approved subsets, which this tokenizer-only
compiler does not infer. The synthetic generator authorizes the stated formatting
policy. The compiler uses at most n(16+1)^2 boundary candidates, with
additional tokenizer costs. Exact tuples, rather than digest equality, decide
admission.

We freeze the contract on the fitting panel before evaluating test names. Its
finite-panel optimality is exact; its admission coverage on future names is an
empirical generalization question. Runtime equality still holds for every
admitted future variant, regardless of whether its name appeared during fitting.

### 3.3 Internal activation isolation and equivalent baseline

Let G gather evidence rows and S scatter them into the original physical slot
layout, with identity positions zero. Native blocks B use evidence-relative
positions and a fixed causal mask. The retained state obeys

    A_0=Embedding(e), A_(l+1)=B_l(A_l), H_l=S(A_l), G S=I.

Equal e gives equal A_0. Induction through deterministic native blocks gives
equal aligned scoring states, final vocabulary arrays and scores. The original
named embeddings are evaluated before the gate, so whole-model identity erasure
is not claimed. Private length, timing and raw embeddings can differ.

An external method passing the same e into the same blocks has the same scoring
property. We test this directly. Deleting name characters and retokenizing the
remaining string is a different encoding: T(prefix+suffix) need not equal
T(prefix)+T(suffix). Internal location alone is not a fairness advantage over
an equivalent external implementation.

### 3.4 Why exact native fidelity can be impossible

For native name scores v_i and an invariant replacement c,

    mean_i (v_i-c)^2 = variance(v) + (mean(v)-c)^2,
    min_c max_i |v_i-c| = (max(v)-min(v))/2.

Thus a name-varying native score cannot be preserved for every name while also
becoming invariant. At a binary threshold, at least the minority count of native
labels must change. These standard identities quantify fidelity, not accuracy.
They do not imply an unavoidable loss against a name-independent correct label.
Our observed credential-accuracy loss remains a real limitation of this model
and readout. Full derivations and boundaries are in paper/THEORY.md.

## 4. Experimental method

### 4.1 Model, data and freeze discipline

We use one frozen affine-four-bit Qwen2.5-1.5B-Instruct derivative on Apple MLX.
The original model and existing measured derivative remain unchanged. Parameter
and file digests are checked. The derivative's source revision and reconstruction
recipe are published; model weights are not bundled.

The public synthetic benchmark contains eight credential task families, two
qualification states and two wording styles, giving thirty-two CV/job conditions.
Six naming variants give 192 planned requests. Labels follow explicit synthetic
credential conditions; they are not expert judgments of real hiring suitability.
Twelve fitting names and twenty-four tokenizer-test names are disjoint. They are
constructed string panels, not probability samples from racial populations.

The tokenizer study uses eight source layouts and enumerates structural formatting
candidates. Its primary contract uses only fitting names. The test-panel optimum
is reported as a descriptive oracle and never selects the primary contract.
That tokenizer study was exploratory. The subsequent GPU source, dependencies,
cohort and endpoints were hashed in a dated local plan before inference. This
is a prospective local freeze, not an externally registered preregistration.

### 4.2 Comparisons and endpoints

The GPU study compares native named scoring, compiled internal routing, external
inference on identical retained IDs, and external character deletion followed
by retokenization. All methods use the same frozen parameters and A/B readout.
No threshold is fitted or selected using the new task labels. A dense physical
and dense logical masking control is retained for the first eight accepted
requests.

Primary endpoints are admission over planned requests, decision-and-score outcome
disagreement, exact score equality and exact aligned-state equality. We compare
all 31 retained scoring stages directly against the same-ID baseline and use
recorded fingerprints for subsequent reproducibility checks. Original native
IDs, full stored slot counts and zero identity rows are also checked.

A separate capability control uses 100 official ARC-Challenge test questions,
rows 200-299, with an explicitly recorded chat prompt and A-D next-letter-logit
readout [12]. We compare every vocabulary logit. This is a standard dataset
subset under our protocol, not a full leaderboard benchmark score or proof of
unchanged general capability.

### 4.3 Statistical and timing units

Names and wording variants repeat the same task families. Task accuracy changes
are aggregated within eight families, followed by a paired family bootstrap
with 4,000 resamples. With only eight constructed families, intervals are
descriptive rather than population hiring inference. No failure-rate confidence
claim treats 192 name variants as independent applicants.

Initial timing used the first eight accepted cases and was poorly balanced
across task families. That result is retained. A separately planned follow-up
balances timing over all eight families using short/long names and randomized
paired repeats. Both internal and external retained-ID services apply the same
authentication, layout, lock and model-guard checks. Compilation and semantic
review are outside the online timing comparison. Other user-authorized training
jobs remain active, so timings concern this host under concurrent load.

## 5. Results

### 5.1 Admission coverage

Across the 192 held-out tokenizer variants, strict boundaries admit 121 and some
permitted token-compatible witness exists for 179. The greedy fitting-anchor
method admits 158; the fitted compiler admits 170, an increase of twelve cases
or 6.25 percentage points. The descriptive test-panel oracle supports 178.

Improvements depend on layout. The tabbed layout changes from 14/24 under the
greedy anchor to 24/24 with the compiler; the space layout changes from 20/24 to
22/24. Other layouts do not improve. Some names remain unsupported. These
results support the feasibility formulation without establishing universal
name admission or demographic admission parity.

The initial comparison changed both contract selection and witness search. A
post-review factorial ablation separates them:

| Fitting contract | First witness only | Exhaustive matching witnesses |
|---|---:|---:|
| Greedy anchor | 158/192 | 160/192 |
| Maximum-support compiler | 154/192 | 170/192 |

With witness search held equal, contract selection adds ten cases on this finite
panel. With the compiled contract fixed, exhaustive matching adds sixteen cases.
The compiled contract with first-witness-only matching is worse than the greedy
pipeline. The full method needs both components; the twelve-case combined gain
must not be attributed solely to contract selection.

### 5.2 GPU conformance and comparisons

All 192 requests in the separate GPU cohort are admitted by both the greedy
anchor and compiler; there is no compiler coverage gain on that cohort. Across
all thirty-two conditions, scores and aligned scoring states are exactly equal
under all six names. No decision-and-score outcome disagreement occurs. The native scorer
changes its binary decision across names in four conditions and has a mean
within-condition logit range of 0.88483.

All 192 external same-ID comparisons match every retained stage. Across direct
counterfactual and reference comparisons, 10,912 arrays match; 5,760 full-shape
zero-identity stage checks pass. External character deletion/retokenization gives
a different evidence sequence and different score in all 192 cases. This
establishes an encoding difference, not that ordinary anonymization is unfair.

Dense logical and dense physical controls show numerical score variation while
the packed reference is constant within a condition. Those controls are small
and descriptive. The study does not establish that a particular fused kernel
algorithm is the sole cause across hardware.

### 5.3 Utility and capability preservation

On the constructed credential tasks, native accuracy is 179/192 (93.23%) and
protected accuracy is 168/192 (87.50%). The paired change is -5.73 percentage
points, with a descriptive family-bootstrap interval of [-14.06,+1.56] points.
A nonsignificant difference is not evidence of noninferiority; no hiring-utility
preservation claim is made. A post-review analysis of the practical
character-deletion baseline finds 168/192 correct as well. Its decisions differ
from the compiled method in twelve cases, despite the same aggregate accuracy,
and from native scoring in twenty-five cases. Thus score/encoding differences
alone do not establish a practical utility advantage for either approach.

On the 100 official ARC-Challenge questions, both paths score 68/100. Every
full-vocabulary array is identical. This is direct preservation on the tested
all-evidence inputs. It does not cancel the credential-task loss or establish
universal preserved performance.

The fidelity audit finds varying native scores in all thirty-two contexts.
At least five binary native decisions must change under any name-invariant
rule on these panels; the implemented route changes fifteen. Thus some changes
are mathematically necessary for native fidelity, but this method is not the
minimum-change invariant rule. Its average best-possible uniform logit fidelity
error floor is 0.44242. Accuracy against the qualification labels is separate.

### 5.4 Balanced timing follow-up

The balanced follow-up contains eighty paired repetitions across eight families
and two names. Internal/external median ratios are first aggregated within family;
the median family ratio is 1.01118. The mean family ratio is 1.01179, with a
descriptive family-bootstrap interval [1.01104,1.01266]. This is about 1.1% online
overhead on this host, with concurrent load and setup/review costs excluded. It
is not evidence that internal routing is faster than equivalent external routing.
The initial dependent-case timing summary remains available but is not used for
this efficiency assessment.

### 5.5 Reproducibility and review status

All public synthetic inputs are generated from published source. Per-case
synthetic records, stage fingerprints, plans, analyses and tests are released.
The benchmark source URL, subset IDs and snapshot hash are recorded. Earlier
private real-CV experiments remain separate supporting studies and are not
represented as independent validation of this new cohort.

Rebuilding the derivative in a new directory reproduces every pinned checkpoint
file and the parameter digest, with all source files unchanged. A fresh-process
repeat preserves all 192 synthetic rows and scoring-stage fingerprints and all
100 ARC records exactly. An initial reconstruction check incorrectly hashed a
model whose arrays had been donated by serialization; offline tokenizer metadata
also differed. The failed check and corrected save/reload procedure are retained.
This is same-host reconstruction, not independent-laboratory replication.

A post-review service check repeats all 192 normal API comparisons against the
actual timed external service and validates an explicit ordered 31-stage trace,
expected shapes, finite arrays and the original fingerprints. This closes a
bridge missing from the initial direct-kernel comparison.

Two fresh-context AI agents reviewed a blinded 24-case annotation packet before
receiving methods or model results. They agreed on all identity spans, retained
spans, qualification statuses and uncertainty decisions; both marked six cases
uncertain. The packet repeats four sentence patterns across six names. Agreement
is automated synthetic consistency evidence, not independent human accuracy.
Their later technical reviews exposed comparator confounding, timing dependence,
a span-fixture offset error, an incomplete cohort audit and semantic-support
wording. The original reviews and response ledger are preserved. Human annotation
reliability, real job-domain labels, a second model family and external-laboratory
replication remain outstanding validation requirements.

## 6. Discussion and implications

The most useful contribution is a falsifiable contract: score equality is
conditioned on exact evidence identity, and rejection is included in the audit.
The compiler supplies a finite feasibility certificate instead of assuming that
whitespace repair makes every native-token input comparable. The strong external
baseline prevents attributing an identical-evidence property to internal location
alone.

For the challenge brief, this addresses behavioral and scoring-activation bias
from designated names: accepted variants have equal score-path states and
outputs. Behavioral equality alone would not imply erasure; a hidden state can
retain a name in a component the output ignores. Our named raw embeddings still
distinguish names, deliberately preserving general model knowledge.

The sociotechnical risks include incorrect role annotation, unequal rejection,
misleading confidence in a synthetic rubric, demographic proxies in retained
qualifications and unfair job requirements. A model can be name-invariant while
making consistently poor or unfair decisions. Human review, appeals and auditing
broader group outcomes remain necessary for a hiring deployment; none is
supplied by the score theorem.

## 7. Limitations and scope of submission

The study uses one checkpoint, one runtime, constructed task labels and a finite
name panel. Its exact theorem depends on correct role review, native-compatible
boundaries, fixed evidence, a 2,048-token input limit and deterministic trusted
execution. It covers scalar scoring rather than unrestricted generated text.
Side channels and raw embeddings remain visible to the trusted host. The
compiler's optimality is finite and policy-relative; future-name acceptance is
not guaranteed. Timing under concurrent load is descriptive.

The proposed mathematical contribution uses elementary established tools. Its
scientific value must be assessed as a focused tokenizer-boundary and audit
study. The work is suitable for external technical review as a transparent
preprint draft; acceptance and venue fit are not established. A stronger hiring
claim requires independent human annotations, credible real task labels and
independent replication. Those cannot be supplied by additional tests on the
same constructed data.

## 8. Availability, ethics and assistance

Code, public synthetic data, audit plans, aggregate and synthetic per-case
records are included in nameguard-peer-review/. No actual applicant records,
learned CV-derived adapter weights, reviewer secrets or base model weights are
published. ARC source material is fetched separately under its dataset terms.
Earlier empirical adapter failures remain available in the repository.

Substantial AI assistance was used in mathematical checking, code generation,
analysis and manuscript drafting. The named author must verify the arguments,
reproduce the experiments, confirm contributions and authorship, and complete
venue-specific disclosures before submission. The blinded AI reviewers are
identified as automated reviewers, not independent human participants.

## References

[1] Robert Wolfe and Aylin Caliskan. 2021. Low Frequency Names Exhibit Bias and Overfitting in Contextualizing Language Models. EMNLP. https://aclanthology.org/2021.emnlp-main.41/

[2] Haozhe An and Rachel Rudinger. 2023. Nichelle and Nancy: The Influence of Demographic Attributes and Tokenization Length on First Name Biases. ACL. https://arxiv.org/abs/2305.16577

[3] Mir Tafseer Nayeem and Davood Rafiei. 2026. Who Gets a Token, and What Does It Carry? Unequal Name Support and Concept Access in Large Language Models. Preprint. https://arxiv.org/abs/2609.34065

[4] Adam Karvonen and Samuel Marks. 2025. Robustly Improving LLM Fairness in Realistic Settings via Interpretability. Preprint v1. https://arxiv.org/html/2506.10922v1

[5] Mark Russinovich. 2026. What the Model Cannot See: Covert and Provable Text Invisibility in LLMs. Preprint v1. https://arxiv.org/html/2610.15883v1

[6] Andrei Sabelfeld and Andrew C. Myers. 2003. Language-Based Information-Flow Security. IEEE JSAC 21(1). https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf

[7] Nora Belrose et al. 2023. LEACE: Perfect linear concept erasure in closed form. NeurIPS. https://arxiv.org/abs/2306.03819

[8] Tim Dettmers et al. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314

[9] Erik Jones et al. 2021. Selective Classification Can Magnify Disparities Across Groups. ICLR. https://arxiv.org/abs/2010.14134

[10] Joshua K. Lee et al. 2021. Fair Selective Classification Via Sufficiency. ICML. https://proceedings.mlr.press/v139/lee21b.html

[11] Sehoon Kim et al. 2022. Learned Token Pruning for Transformers. KDD. https://arxiv.org/abs/2107.00910

[12] Peter Clark et al. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
