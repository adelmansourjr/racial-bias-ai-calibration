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
