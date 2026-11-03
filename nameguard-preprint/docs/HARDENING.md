# Specific trade-offs and remaining weaknesses

| Issue | Implemented measure | Remaining limitation |
|---|---|---|
| Scalar token explosion | Tokenize one canonical evidence sequence inside the model; transformer IDs match ordinary evidence BPE exactly. | Original source is also embedded; total embedding visits and CPU review are extra work. |
| Evidence-field BPE boundary leakage | Concatenate evidence before its one tokenizer call. | This is explicit internal text canonicalization, not autonomous learned unlearning or a tokenizer-free activation projection. |
| Whole-source BPE entangles name and fact tokens | Original source embeddings exist first; initialize the canonical branch from native evidence embeddings independently. | Raw embeddings/name length remain privately observable; the source path is not semantically erased. |
| Missing registered body reference | Source-global normalized registry coverage rejects any unexplained match. | Unknown aliases, undeclared transliterations, semantic coreferences and unsupported homoglyphs can remain. |
| Name equals skill or client | Source-bound retained role, exact text/offsets and reason required; overlaps reject. | Reviewers can still approve a false retained meaning. |
| Incorrect or conflicting review | Two distinct credential keys must endorse the same resolved source-bound payload; reject disagreement and unresolved proposals. | Credentials do not prove two humans or independent judgments. Shared semantic errors are possible. |
| Mutation/tampering | Immutable bytes, duplicate-key rejection, strict schema, source hash and HMAC payloads. | Trusted key holders can issue wrong reviews. HMAC is not asymmetric non-repudiation. |
| Model/cache interference | Pinned files/parameter digest, per-call graph/object guards, no external caches, global model lock and poison-on-backend-failure. | Benign operator/runtime and immutable arrays remain trusted. Not hardware attestation or a hostile-host sandbox. |
| Annotation/BPE limits | Original-source and evidence-token limits reject instead of truncating. | A canonical evidence overflow occurs inside the backend and poisons that service; rebuild before reuse. Admission/timing equality is not claimed. |
| Confidence as hiring probability | Public output is explicitly constrained A/B model score. | No calibrated real hiring probability or valid job rubric is established. |
| Benchmark overclaim | Frozen exact token/tensor checks and 21 warm timing pairs, with 32 generated token-corpus cases. | Synthetic repeated measures and one local checkpoint; not a population study or multi-device speed guarantee. |
| Grading/performance ambiguity | Native evidence hidden states and same-convention readout match exactly. | Native named-CV outputs and real hiring accuracy can differ. We do not reinterpret token parity as all-capability evaluation. |
| Claim outside accepted fields | Precise theorem assumes unchanged canonical evidence. | Other proxies, population correlations, runtime side channels, external name-based ranking and all-name semantic discovery remain outside it. |

## Why we did not claim a universal role detector

The same phrase can denote an applicant or a third party. Observationally
identical text can require different labels under different outside contexts.
No signature, disagreement coefficient, finite benchmark or lexical inventory
supplies that missing context automatically. We preserve an unregistered-alias
negative control rather than silently claiming it solved.

The correct operational response to ambiguity is review/adjudication or refusal
to score. A future real-review study should estimate omissions and false removal
against independent ground truth. The current exact annotations come from a
controlled generator, and the two signing credentials are simulated fixtures.

## Research-claim hardening in v0.3.0

The literature and CLAIM-EVIDENCE.md narrow unsupported claims. Eleven new CPU
checks validate exact counterexamples and citation consistency. Equivalent
external-Q routing has the same output guarantee; raw embedding first adds no
score-isolation benefit. No learned-erasure, human accuracy, global utility or
novelty result follows. The frozen experiment is unchanged; this is reporting
and evidence hardening rather than an unmeasured model fix.
