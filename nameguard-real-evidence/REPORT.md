# Exact name invariance and the limits of verifying evidence in real CV text

*Research addendum, 16 November 2026*

## Research question and outcome

The previous controlled study recovered definitive coverage using a trusted
credential registry. Here the scorer receives **real CV text only**, with no
issuer register, verified identity binding or authoritative event dates.
We ask whether a short excerpt explicitly reports the applicant personally
using a named tool in work or a project. We measure source-span precision
and coverage separately from name invariance.

Across **119 distinct real CV excerpts**, changing the designated applicant
name changed native model decisions in **18** conditions. The same change
altered **zero** protected score vectors or checked explanations. All **476**
planned named requests were admitted, and all **1,488** sampled aligned
activation arrays matched bit for bit.

The automated verifier did **not** reach the joint target of zero unsupported
confirmations and 60% recall on the held-out cohorts. Some emitted quotes
were verbatim, but did not establish that the applicant personally used the
tool. This negative result is part of the contribution.

## What a source check can establish

1. **Provenance.** A citation contains start and end character offsets and
   the SHA-256 of the corresponding substring in the whitespace-normalized
   excerpt. A deterministic verifier can reconstruct the quote and reject an
   incorrect or modified span. Every emitted citation passed this check.
2. **Attribution and entailment.** A quotation supports the chosen claim
   only if it assigns tool use to the applicant in the relevant context.
   Disjunctive list headers, passive voice, proficiency claims, future
   intentions and third-party activity can all make an exact citation
   insufficient. The grammar recognizes some of these distinctions and has
   measured errors on the rest.
3. **Truth in the world.** The CV does not independently authenticate its
   claims. Two applicants could present identical text even if only one used
   the tool. A system reading the text alone must return the same result in
   both worlds and cannot correctly certify both. Reliable credential or
   employment verification requires independent evidence beyond the CV.

This separates traceable output from semantic evidence. The experiment pairs
the source-span design with questions that have misleading literal mentions,
following prior work on fact verification, answerability and citation quality.
[FEVER](https://aclanthology.org/N18-1074/) ·
[SQuAD 2.0](https://arxiv.org/abs/1806.03822) ·
[ALCE](https://aclanthology.org/2023.emnlp-main.398/)

## Data, reference labels, and freeze order

Use sampled windows in the public [Kaggle résumé dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
Keep the tool mention, surrounding prose and original tokenization after
whitespace normalization. Exclude contact-like snippets, control delimiters,
duplicate normalized window text and recoverable prior-study IDs or prefixes.
The 119 evaluated excerpts are distinct from the 12 development CVs by row ID,
normalized prefix and sampled window. The 119 examples are selected to contain
a tool mention, with and without a nearby lexical activity cue. Their frequencies
do not estimate how often an arbitrary CV supports a qualification.

The first holdout has 48 documents spanning Excel, SQL, Python, AutoCAD,
Java and SAP. An initial conservative grammar was frozen before reading those
test excerpts. The first revision addresses structures missed in that cohort
and was frozen before sampling a separate 35-document cohort. There were
too few disjoint Python near-verb records to fill the planned strata. Two
failed sample configurations and the final 35-case rule are recorded. No
earlier fresh cohort was written. The next revision guards the attribution
failures discovered in the 35-case audit. Its confirmation cohort has 36 new
documents drawn from tool families never used to develop the grammar: Tableau,
MATLAB, QuickBooks, WordPress, Photoshop and Salesforce. The first sampling
attempt found no new qualifying Revit near-verb examples. Photoshop was
substituted before the confirmation excerpts and labels were available.

A single AI research annotator reviewed the excerpts and recorded a verbatim
positive citation or a review label before each GPU run. The annotator did not
see model outputs or predictions for the upcoming split when labeling it.
Because that annotator also designed the study, these are reference judgments
and not independent human gold. Review and support require interpretation.
For example, generic experience without a stated task was conservatively
labeled review. SQL database or SQL-family language use qualifies for the
SQL question. Counts are conditional on these rules.

All source data, private annotations, quotes, case-to-row maps, offsets and
per-CV outputs stay local. The package includes competing pipelines,
public dataset fingerprint, frozen source hashes, aggregate reports and
synthetic tests. It cannot provide third parties the exact privately labeled
cases for independent semantic adjudication from the public repository alone.

## Inference and exact checks

Use a pinned frozen 4-bit Qwen2.5-1.5B-Instruct derivative on Apple MLX. Each
excerpt receives the same A/B question across four new intervention strings
that were absent from this project's prior experiments and the fixed role
witness panel. Pretraining exposure to those strings is unknown. The original
named prompt enters tokenization and embedding. The protected path zeroes the
designated identity rows and executes native blocks on the retained evidence,
using identical positions and masks on that active sequence.

For two admitted source variants with the same retained evidence token tuple,
the evidence embeddings are equal. Each native block receives the same
evidence and execution inputs. Induction gives equal aligned states and
score vectors in the deterministic runtime. The source verifier uses the same
excerpt and exact offsets for both variants. Its output serialization is
therefore the same. This proof does not cover unannotated identity elsewhere
in the text, changes to retrieved evidence upstream, admission from arbitrary
boundary layouts or timing. We count all planned requests against admission
and explicitly measure output and activation equality.

The checked path makes the documentary decision from the span rule. Qwen
scores still travel the protected path and are audited, but cannot replace
missing evidence. Rule-only decisions match by design. The optional comparison
requires both a checked span and a model A preference. We give performance
credit to the verifier only where the reference label agrees.

## Results

| Holdout and frozen method | Excerpts | Labeled supported | Correct supported | Unsupported confirmations | Positive recall | Definitive coverage |
|---|---:|---:|---:|---:|---:|---:|
| Initial, conservative grammar | 48 | 17 | 7 | 0 | 41.2% | 14.6% |
| Fresh documents, expanded grammar | 35 | 12 | 5 | 2 | 41.7% | 20.0% |
| New documents and tool families, attribution guards | 36 | 7 | 4 | 2 | 57.1% | 16.7% |

Each cohort evaluates a different frozen revision. These rows describe measured
replication and revision performance, not a pooled causal comparison of rules
on the same inputs. In the final cohort, protected Qwen alone supported three
correct and six unsupported claims. Requiring both the final grammar and Qwen
agreement supported two correct claims, emitted zero unsupported claims and
recovered only **2/7** positives. This distinction prevents an almost-always
reviewing method from passing as useful. In the 35-case cohort, agreement still
emitted two unsupported claims.

All accepted citations were exact source spans. All sixteen supported decisions
credited by the reference judgment across the three frozen runs overlapped
their reference citation (7 + 5 + 4). Exact location did not prevent four
unsupported citations. Two were ambiguous use or attribution in the 35-case
cohort. The final holdout added generic experience wording and passive project
description with no confirmed author role.

| Exactness endpoint | Result |
|---|---:|
| Named requests admitted | 476/476 |
| Conditions with identical protected logits and report bytes | 119/119 |
| Native score-vector changes across names | 119/119 |
| Native binary model decision changes across names | 18/119 |
| Bit-exact aligned array comparisons | 1,488/1,488 |
| Designated identity zero checks | 1,920 passed |

On every eighth document, compare the 31 aligned stages, including the final
vocabulary array, across three name variants against the first name. Verify
all model weights after inference. These are direct conformance counts.
The four name variants do not count as four utility samples.

The additional actor guard routes generic experience and unnamed passive action
to review. Applied **after seeing outcomes** to the existing excerpts, it
supports 27/36 positive cases with zero unsupported confirmations. On the
36-case confirmation subset it supports 4/7 with zero unsupported
confirmations. This reanalysis diagnoses the two newly observed error
mechanisms. It has no fresh holdout and did not trigger a new GPU run. The
zero error count applies only to the already inspected material.

## Preprint claim and remaining gates

The combined project is suitable as a focused preprint draft on **exact
name-channel noninterference and the limits of evidence-grounded CV scoring**.
It offers a formal conditional property, measured real-text conformance,
an explicit distinction between source integrity and semantic support, and
a reproducible negative result on joint extraction safety and coverage. The
controlled registry experiment answers a different question and can remain
as a contrasting input contract.

The present results cannot support a claim of reliably safe automatic
credential extraction, validated hiring utility or population racial
fairness. An extended paper should obtain independent annotations from at
least two blinded human reviewers, document disagreements and adjudication,
use a larger reserved CV corpus with independently sampled linguistic
patterns, report coverage at a declared risk target, and enable independent
replication with an agreed privacy-preserving evaluation set. Exact score
invariance is already a proved conditional property. More selective grammar
design and human semantic verification address a different failure mode.
