# Real CV document-evidence protocol

## Question

Does this CV excerpt explicitly report that the applicant used the named tool
in work or a project? This is a documentary claim. A checkable excerpt cannot
establish that the experience really occurred, that a credential is genuine,
or that a person should be hired.

Mark **supported** when the excerpt links the CV author to actual tool use in
a stated task or hands-on experience. Mark **review** when it only lists the
tool, mentions training or knowledge, describes another person's use, gives
a requirement or future plan, denies use, or is truncated or ambiguous.
Technical responsibilities phrased in ordinary CV shorthand count if the
actor and the activity-to-tool link are clear. Evidence must be a verbatim
quote from the supplied excerpt. Review is never an automatic rejection.

## Data and measurement

Use six named tools, two lexical sampling strata per tool, one distinct real
CV per stratum for development, and four additional distinct CVs per stratum
for the held-out run. Source excerpts are whitespace-normalized windows centered
on a sampled tool mention, with attributable evidence span coordinates in that
normalized source. Both strata contain the tool. Sample choice uses neither
model scores nor the final evidence rule. For this run, exclude recoverable
prior experiment IDs and matching prefixes, then remove within-run duplicate
IDs, excerpts and prefixes. That is a reproducible effort to reduce overlap,
not a guarantee of pretraining novelty. The sample is enriched for lexical
mentions and does not estimate coverage over arbitrary CVs or real applicants.

One researcher labels the 48 test excerpts before model inference. A
positive label provides an exact quote and an annotator note. This is an
output-blinded, single-annotator reference set, not independent human gold.
Record each unsupported emission, review decision and supported-span
disagreement. Test the correctness of every positive output against these
labels, distinguish exact span provenance from semantic agreement, and
count rejected name variants explicitly. Report positive recall and overall
definitive coverage separately. The local working target is zero unsupported
definitive decisions with at least 60% recall of labeled positive excerpts.
Keep all results if a target fails.

All held-out excerpts cross four previously unused intervention names in the
designated identical role slot. The original named prompt reaches tokenization
and embedding. The role contract fixes the retained evidence. Compare the
two option logits bit-exactly and the report bytes across the four names.
Capture all aligned stages on every eighth held-out case. Each real CV is
counted once for utility. The name replicas are conformance checks.

## Comparison arms

- **Mention only** emits a positive result whenever the tool occurs. This
  stress-cohort baseline treats every sampled mention as decisive.
- **Native Qwen** gets the entire named prompt and reports the binary choice.
- **Protected Qwen** uses the same prompt and exact evidence routing.
- **Protected Qwen with source verification** emits the document-supported
  result when there is a checked span, otherwise refers. Qwen values still
  get audited, and model-only positives cannot override a missing witness.
- **Verifier alone** takes the same excerpt with no model input. The verified
  report is defined by the span rule, so equality between these latter two
  arms is expected. We also report requiring both model and rule to accept,
  since agreement can change coverage even when output safety is identical.

The verifier checks a bounded language pattern and matching source offsets.
It is a selective evidence rule, not an unrestricted semantic proof. Ambiguity
and conflicting cues matter. Keep candidate names, source excerpts, offsets
and per-CV records local. Publish code, controls and aggregates only.

Inspiration includes evidence-sensitive evaluation and deliberately
unanswerable controls. [FEVER](https://aclanthology.org/N18-1074/) ·
[SQuAD 2.0](https://arxiv.org/abs/1806.03822) ·
[ALCE](https://aclanthology.org/2023.emnlp-main.398/)
