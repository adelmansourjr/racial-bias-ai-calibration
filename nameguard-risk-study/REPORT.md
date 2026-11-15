# Exact name-field noninterference and selective source verification

**Research report and preprint draft**

**Abstract.** An evidence scorer can be exactly invariant to a declared name
field and still make unsupported claims. We test these properties separately
with a frozen Qwen2.5-1.5B-Instruct checkpoint, internal activation routing,
and human-annotated evidence. A candidate-aware verifier fails its development
criterion despite discriminating positives from plausible distractors above
chance. A separately frozen follow-up first extracts an answer without seeing
the proposed candidate and then checks exact agreement and source offsets.
Across 320 reserve cases it made 55 confirmations, including 19 unsupported
claims, and missed the joint risk and coverage gate. All 20 four-name generation
orbits matched exactly, with 375 direct full-vocabulary comparisons. This is
evidence that exact name noninterference extends to autoregressive evidence
generation and can coexist with substantial semantic error. Both protocols, public case IDs, executable analyses, and
model fingerprints are released.

## 1. Research problem

A résumé scorer may treat a name as evidence, causing decisions to change when
qualifications remain fixed. A second, logically independent error occurs when
it attributes a real quotation to the wrong person, activity, or criterion.
In the [preceding real-CV study](../nameguard-real-evidence/REPORT.md), protected
scores and reports were exactly invariant in 119 matched groups. Native binary
choices changed in 18. In the final 36-excerpt holdout, two of the six reported
tool-use confirmations were unsupported under the study rubric. The more
conservative agreement rule emitted zero unsupported confirmations in that
cohort and recovered only two of seven positives. That result relied on one
blinded AI reference annotator. The present experiment adds independent human
labels on a different task with a harder literal-span control.

The main contribution is a test of how **exact information-flow isolation
composes with usable evidence verification**, with three distinct observations.

1. We test exact equality along the model path and at the report seen by users.
2. We compare a candidate-aware yes/no classifier and a candidate-blind
   extractor on the same human-annotated adversarial evidence.
3. We evaluate a predeclared risk–coverage gate, with an exact finite audit on
   accepted claims and a separate exhaustive census.

## 2. Related work and contribution

**Matched name interventions.** Bertrand and Mullainathan (2004) randomized
names on otherwise matched résumés for a labor-market correspondence
experiment. Wilson and Caliskan (2024) adapted this approach to embedding
retrieval in résumé screening. A paired intervention here estimates sensitivity
to a substituted string on fixed evidence. Name strings supply counterfactual
test inputs without demographic labels or claims about an individual's race.
[Bertrand and Mullainathan, 2004](https://doi.org/10.1257/0002828042002561) ·
[Wilson and Caliskan, 2024](https://arxiv.org/abs/2407.20371)

**Representation interventions.** LEACE (Belrose et al., 2023) gives a
closed-form guarantee against linear decoding under its assumptions. The
present proof concerns a specific token channel and exact computation on
pairs, regardless of the downstream function's linearity. The raw
embeddings still encode names. The accompanying
[same-ID reference and method comparison](../nameguard-peer-review/paper/manuscript.md)
analyzes how an external path supplied the identical retained token IDs can
implement the same function. The main engineering contribution of the internal
path is compatibility with original named input and inspectable isolation at
the embedding boundary. It uses established ideas from information-flow
noninterference. [Goguen and Meseguer, 1982](https://doi.org/10.1109/SP.1982.10014) ·
[Belrose et al., 2023](https://arxiv.org/abs/2306.03819)

**Evidence verification.** QAGS and QAFactEval use question answering to
compare a claim with its source. We adapt that idea to supplied candidates
and assess exact reproduction on SQuAD 2.0, whose authors deliberately wrote
unanswerable questions with plausible distractors. Permuting response labels
in the direct verifier controls for one type of answer-label preference.
Comparing on fixed cases helps distinguish improved evidence behavior from
changes in case mix or prompt option placement.
[Wang et al., 2020](https://aclanthology.org/2020.acl-main.450/) ·
[Fabbri et al., 2022](https://arxiv.org/abs/2112.08542) ·
[Rajpurkar et al., 2018](https://arxiv.org/abs/1806.03822) ·
[Tang et al., 2024](https://arxiv.org/abs/2310.07712)

**Risk and review.** Selective prediction evaluates error against the
proportion of inputs on which a system acts. Learn then Test gives a general
framework for finite-sample risk control with held-out calibration. This study
uses an exact finite-population audit because the accepted set is fixed and
there is no basis to treat Wikipedia cases as independently sampled from future
job applicants. Exact inversion of a hypergeometric tail is standard.
[Geifman and El-Yaniv, 2017](https://arxiv.org/abs/1705.08500) ·
[Angelopoulos et al., 2021](https://arxiv.org/abs/2110.01052)

## 3. Methods

### 3.1 Cases and comparisons

SQuAD 2.0 development release, pinned by SHA-256. For each eligible paragraph,
choose one exact, source-grounded official answer and one official plausible
distractor whose text also occurs at its labelled offset. These are attached to
different questions in the same paragraph. A plain string-presence check
therefore accepts both. Any policy using only the fact that the quote occurs at
the asserted offset sees the same feature on every item. A deterministic
policy based on that feature alone must choose between zero coverage and the
50% error rate of this balanced benchmark. Freeze whole articles into eight development articles,
17 first-study held-out articles and ten reserve articles. The three samples
contain 144, 312 and 320 questions, respectively. The local source manifests record each protocol and cohort before the corresponding run.
They are reproducibility records, without an independent timestamp or registry.
The follow-up protocol
was frozen after discovering the development failure and before any inference
on reserve articles. It uses the same cases for both verifier methods.

The first classifier scores the logit difference for A and B using both answer
mappings. If (s_A = l_A-l_B) when A means support and
(s_B = l_B-l_A) when B means support, the mean
((s_A+s_B)/2) removes any *additive mapping-independent* label logit
bias. This arithmetic does not cancel content-dependent mapping effects.
The minimum rule requires both signed scores to clear the same threshold.
Each cutoff is chosen entirely from development observations, with a minimum
of 12 accepts and a maximum of 4% empirical false support. If no cutoff
qualifies, the operational rule refers all cases. AUC still records ranking
performance independently of that operating point.

The second method makes one greedy answer sequence (u) from only the
passage and question. Let (g(u)) return a terminated single line with
whitespace and paired quotation marks stripped. Let (a) be the proposed
answer and (p[i:j]) be its prescribed source slice. The verifier emits a
positive report only if (g(u)=a=p[i:j]) and the output is neither blank nor
the cannot-answer sentinel. The raw output never determines source offsets.
All other cases are referred. Exact reproduction is intentionally conservative.
Paraphrases, punctuation changes and equivalent longer spans can cause referral.

### 3.2 Conditional proof

Let (x) be full original token IDs for a reviewed named input, and (P) the
ordered tuple of its retained evidence positions. The runtime evaluates the
full lookup (E(x)) and gathers (E(x)_P) before attention. The role contract
fixes (x_P) across admissible counterfactual variants. If (x_P=x'_{P'}),
then their gathered embeddings are identical. For the cached extractor, the
same native blocks, logical positions, masks and empty per-request caches
receive these equal arrays. Under deterministic execution they compute equal
prefill states, next-token logits, and caches. Greedy decoding then produces
the same next token in both runs and feeds it back at the same logical
position. Induction to the common stopping rule gives equality of the whole
token sequence. Checking exact source and candidate against the same reference
yields identical reports.

This proof assumes correct role spans, fixed retained IDs and evidence,
deterministic kernels and a fixed model and response policy. The observations
check this implementation on a pinned runtime. Name changes may affect total
tokens or fail the tokenizer or review contract. The variables in the proof
exclude raw identity embeddings, private sequence geometry, other names
embedded in evidence, timing and externally logged identity. A change to
the rubric, retained evidence, or review outcome is a different intervention.
Equal outputs alone would not establish this property. For example, a hidden
state \(H=(V,N)\) can retain the name \(N\) perfectly while a readout
\(f(H)=V\) is invariant. The experiment samples internal and external
observables because those are different claims.

The routing leaves original input IDs intact. With \(L\) retained prompt
positions and \(t\) generated positions, cached attention uses
\(O(L^2+tL+t^2)\) arithmetic across the prefix and decode, omitting fixed
model dimensions. Identity slots add embedding and routing overhead but do not
add attention positions. This is an operation count. Kernel launch costs,
linear projections and retrieval work also affect end-to-end runtime.

### 3.3 Exact finite-set audit

Let (A) be the frozen set of accepted cases, (N=|A|), and (K) the number
of unsupported accepts. Before inference we commit a uniform random permutation
of all holdout IDs. After freezing results, restricting that order to (A)
gives a uniform permutation of (A), since every relative order of a fixed
subset appears equally often. Its first (m=min(160,lceil0.8N
ceil)) elements
are a simple random sample without replacement. If the sample has (x) false
supports, the one-sided 95% upper bound is

```
U = max { K in [x, N-m+x] : P_K(X <= x) > 0.05 }

P_K(X <= x) = sum_{i <= x} C(K,i) C(N-K,m-i) / C(N,m)
```

Use only feasible terms in the sum. For a fixed (K), the probability of
(P_K(Xle X_{
m obs})le0.05) is at most 0.05 by discreteness. The tail is
nonincreasing in (K), by coupling error sets with one set a superset of the
other. Thus inversion gives exact conservative coverage and permits binary
search. Integer arithmetic avoids numerical tail error. On an empty accepted
set the audit gives no useful certification. The full benchmark is labelled,
so the sample estimates an already knowable count. It tests the audit procedure
as if only a subset could be reviewed, and the full census checks it. With
sampling by question ID, within-paragraph or within-article dependence of
errors does not invalidate this *conditional finite-set* statement.

For a name orbit that satisfies the proof, repeated variants share the same
acceptance, report and ground-truth evidence label. Auditing once per orbit
estimates error among base claims, with zero statistical precision gained from
counting repetitions. The guarantee does not generalize the labelled census
to future CVs. The predeclared joint gate requires at least 20% coverage, at
most a 10% upper audit error rate, and all planned exact-name checks to pass.

## 4. Results

### 4.1 Candidate-aware verification failed its registered operating point

No development cutoff satisfied both the minimum acceptance count and maximum empirical
error. Even the 20 highest mean-margin development scores included three
unsupported candidates. Every frozen rule therefore referred every held-out case.
All 312 inputs passed the tokenizer and role contract. On those 312 cases,
the single, mean, and minimum margins yielded AUROCs of **0.790, 0.761,
and 0.769**, respectively. Ranking skill did not establish a useful operating
point. The additive-bias control showed no AUC gain on this holdout.

On the 20 preselected name orbits, all four names produced identical A/B
logit bits in both mappings, identical report digests and matching aligned
activation digests. All **3,720** cross-name stage digests matched, with
**4,800** zero-row checks in the protected full layout. The unrestricted
first mapping's logits changed in all 20 orbits, and its binary choice
changed in one. This choice comparison is unthresholded. The all-review
deployment rule trivially makes the same decision across names, so the
logit and activation checks are essential.

### 4.2 Candidate-blind verification still failed the joint gate

All 320 reserve cases passed admission. The candidate-blind method made
**55 confirmations (17.19% coverage)**, of which **36 were supported** and
**19 unsupported (34.55% error)**. It recovered 36 of 160 supported candidates.
The held-out audit found 15 unsupported confirmations among 44 audited
accepts and gave a one-sided 95% upper count of **22/55 (40.0%)** in the
finite accepted set. The exhaustive error count of 19 was below that bound.
Both thresholds missed their targets. The exact-name gate passed.

| Method on the same 320 reserve cases | Confirmed | Supported | Unsupported | Error among confirmed |
| --- | ---: | ---: | ---: | ---: |
| Literal span at supplied offset | 320 | 160 | 160 | 50.00% |
| Registered candidate-aware mean rule | 0 | 0 | 0 | Undefined |
| Candidate-blind exact reproduction | 55 | 36 | 19 | 34.55% |
| Candidate-aware mean ranking at 55 cases, post hoc | 55 | 44 | 11 | 20.00% |

The candidate-aware mean ranking had reserve AUC **0.727**. Its top-55 comparison
is retrospective and would still miss the 10% target. No adjusted threshold is
promoted to a validated result. Removing the candidate from the extraction
prompt did not reliably distinguish supported evidence from a plausible
distractor at the same coverage.

A post-hoc author inspection of the 19 errors found several concrete failure
modes. The text attributes the historical appointment to a successor, but
the question asks about the predecessor. A passage reports the busiest airport
while the question asks for the second busiest. Two population-based questions
contain near-matching digit sequences that the model treats as identical.
Other cases reverse negation or relational direction. Exact answer strings
therefore mask person binding, numeric fidelity, and scope errors. Some
official negative examples use a likely typo or reversed name order.
Even reclassifying two such examples as supported leaves 17 unsupported
accepts of 55 (30.91%). This sensitivity check is descriptive and uses the
original labels for the registered analysis.

The 20 protected generation orbits all matched across four names. We compared
the packed embedding prefix, generated token IDs, report bytes, and the raw
bytes of every observed vocabulary array directly. All **375 cross-name
vocabulary arrays** matched. Each accepted report also reconstructed from
the official source offset on the CPU. The checkpoint and parameter digest
matched in both studies. These are exact execution results, even though
semantic correctness was substantially lower.

### 4.3 Diagnostic controls

Two controls were specified after the reserve results and run on those same cases.

**Same-ID reference.** The native model saw exactly the ordered evidence IDs
retained inside the protected path. Across all 20 name orbits, the initial
packed embedding digest, generated token sequence digest and termination matched
between the ordinary retained-ID reference and the protected path. All **500**
vocabulary-step digests matched across the four protected variants and the
one reference per case. When the native model instead received the full
named input, its generated token sequence varied across names in **7/20**
orbits. Exact confirmation itself did not vary in that small control panel.
The equivalence shows that internal isolation reproduces the computation
with identical evidence IDs, so an external implementation with those IDs
is an equally strong functional baseline for these outputs.

**Blank-passage ablation.** Compare the same questions with the passage empty,
scoring the raw answer match before source-offset verification. Blank input
gave a matching answer in **1/320** cases, which was also supported and
matched with the full passage. With the passage present, **55/320** answers
matched. All **19 unsupported confirmations** disappeared when the passage
was removed. The full source therefore changes the answer even when the
conclusion is wrong. The result is consistent with incorrect relational or
numeric reasoning on available evidence. A blank passage changes prompt
structure and cannot rule out pretraining recall or quantify the causal
effect of any individual sentence.

## 5. Validity, use, and preprint scope

The benchmark balance is intentionally 50/50. It supplies no estimate of
prevalence in job applications. If sensitivity \(t\) and false positive rate
\(f\) transferred unchanged to a population with prevalence \(p\), false support
among accepts would be \(f(1-p)/(tp+f(1-p))\). The release reports that
sensitivity analysis using the observed class-specific rates. There is no evidence
that those rates transfer to CVs.
Article separation protects against adaptation to observed labels on these
articles. The public release may still occur in pretraining. A plausible
answer is considered false under the official task label. Some questions
may admit reasonable alternative readings and the task does not verify an
assertion's real-world truth. A model could retrieve a known answer from memory.
An exact quote verifies provenance. Source support and factual truth require
different labels and tests.

Real applicants need a clear explanation of the decisive evidence, a way to
correct extraction or annotation errors, and a path to review. A name field can
be blocked while other cues in the retained record affect the result. Different
naming conventions may face different admission or referral rates. Those rates
and rates of false confirmation need separately labelled checks. Four
experimental names establish conformance for this panel and support the
conditional proof. They do not measure demographic parity or represent all
names. Using the benchmark result for automated hiring eligibility would
exceed the experiment's scope. Registry-backed credential validity and
benchmark source support both differ from whether a résumé statement should
determine a person's suitability for a particular job.

The defensible paper claim is deliberately narrow. A precisely scoped scoring
and generation path has a conditional noninterference proof with exact runtime
checks. The source-verification experiments can succeed or fail independently.
The proof, controls and observed failures support a methods and evaluation paper. Fresh
human-reviewed CV validation, an independent checkpoint or runtime replication,
and review of tokenizer admission across naming conventions are needed for
general hiring or broad model claims. The release is a focused technical preprint draft. The originality claim is
the joint experiment and the application of exact routing to generated evidence.
A peer-reviewed venue would still need to judge significance and external validity.
