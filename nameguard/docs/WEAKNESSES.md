# Specific weaknesses, fixes and residual limits

This ledger audits `universal-name-noninterference`, not the earlier learned
adapter as if it had acquired a universal raw-text guarantee. Severity refers
to the intended score noninterference claim, not a generic security rating.

| ID | Concrete failure or missing premise | Change and evidence | Remaining boundary |
|---|---|---|---|
| W01 | Applicant/caller can relabel an identity node as evidence after review. The old API accepts unauthenticated role dictionaries. | HMAC-SHA256 binds the exact role/text nodes, source hash, reviewer and policy. Five real request mutations are rejected before GPU work; CPU wrong-key/unknown-reviewer tests pass. | Reviewer/key-holder is trusted. Authentication does not validate semantics or prove an actual human reviewed the source. |
| W02 | The same retained text can tokenize differently when evidence-field boundaries change. Four actual-tokenizer regression examples demonstrate this. | Encode every original Unicode scalar separately, including identity text. This homomorphism makes retained token IDs independent of field fragmentation. Actual GPU variants and 2,000 seeded Unicode property cases agree. | This differs from ordinary whole-string BPE and can degrade task utility. NFC/NFD changes to retained evidence are not declared equivalent. |
| W03 | An `audit=True` option returns identity-dependent embedding hashes and source length. Those outputs violate a whole-response noninterference claim. | Serving returns only an immutable `PublicScore` with two fixed scalar fields. No serving audit argument exists. Tensor observations use a trusted experimental backend subclass. | Private experimental traces, host memory and runtime timing are not protected by this output API. |
| W04 | Mutable nodes can change after validation; duplicate JSON keys make signed interpretations ambiguous. | Requests are bounded immutable bytes; duplicate keys are rejected; parsed roles become frozen node tuples. Signing snapshots the source. | Arbitrary malicious code in the host process remains outside the threat model. |
| W05 | A cache, another overlay or a swapped model module can invalidate the proof. | No external cache argument; fresh prefill only; pinned parameter digest and architecture; per-call module/tensor-object fingerprint; overlay flag rejection; final parameter and checkpoint-file rechecks. | Immutable-MLX-array assumptions and a benign host are trusted. Content hashing every parameter on every call is not performed. |
| W06 | Per-service locks alone do not serialize multiple wrappers sharing a model. | The supported factory uses a process-wide nonblocking inference lock in addition to the service lock. Actual smoke test verifies rejection. | This intentionally rejects overlap; it is not a multiuser server or protection against independent malicious processes. A busy backend rejection conservatively poisons that service. |
| W07 | A backend exception may leave state uncertain and leak raw details through an error message. | Backend failure becomes a generic error, disables that service and releases locks in `finally`. Tests cover failure, recovery refusal, close and concurrent access. | Operators must rebuild a failed runtime. Admission errors are still observable; no termination-sensitive guarantee is claimed. |
| W08 | Matching activation hashes is evidence, not direct array equality. | New GPU study compares NumPy arrays element-for-element at all 31 public stages on each forward, plus the original Qwen path on the same scalarwise evidence. | Saved metadata reconstructs coverage but does not contain all large activation arrays. The CPU verifier is not independent GPU re-inference. |
| W09 | The solver result checks a one-block abstract lemma, not complete code. | Add a symbolic two-identity-block lemma and an unsafe-position countermodel. Both safe lemmas are UNSAT; finite-test and original-position counterexamples are SAT. | No end-to-end mechanized proof of Python, tokenizer, MLX, kernels or review semantics. Do not call this verified deployment software. |
| W10 | A name-based tie-break can undo invariant individual scores. | Provide a ranking helper using fixed operator-assigned integer slots; reject duplicate/invalid slots. | Slot assignment itself must be name-independent. An external application can still bypass this helper. |
| W11 | Large inputs, unsupported Unicode or empty evidence can fail in identity-dependent ways. | Bound source/identity bytes, nodes and token counts; reject malformed UTF-8/surrogates and silent truncation; test limits. | The theorem compares accepted requests, not rejection behavior, latency, memory use or denial of service. |
| W12 | Finite handpicked names can be mistaken for a universal proof or population study. | Separate solver lemmas, handwritten induction, implementation conformance, deterministic fuzzing and unresolved semantic premises. Publish full factorial coverage and exact scope. | Three synthetic evidence bodies are not independent population hiring samples. No binomial confidence claim is attached to the 108 repeated forwards. |
| W13 | “Base unchanged” can be misreported as unchanged grading accuracy. | Explicitly disclose the new scalarwise input convention, original model digest and token counts: 59/60/61 ordinary BPE versus 271/281/274 scalar tokens. | No hiring accuracy, native raw-CV output preservation or latency parity is established. |
| W14 | General name knowledge or other racial proxies may remain despite name-channel isolation. | Retain the narrow theorem: designated-identity intervention cannot change accepted public scores with fixed evidence. Deliberately preserve a bad-review negative control. | Unknown aliases, wrong semantic roles, schools, geography, job policy and group outcome fairness remain unresolved. |
| W15 | An anonymous proof artifact is not automatically publication-ready. | Add a reproducible package, threat model, artifact manifest, failed-baseline provenance, machine-checkable specifications, CI recipes, manuscript source, results and limitations. | This is a preprint draft/research prototype. Independent replication, academic novelty, real reviewer study and external peer review are absent. |

## Research basis

The design distinguishes confidentiality (identity cannot reach scoring) from
integrity (untrusted requests cannot alter reviewed roles). This distinction
and a clearly named trusted computing base are standard in information-flow
systems; see [Chong, Vikram and Myers, SIF](https://www.cs.cornell.edu/andru/papers/usenix07-html/paper.html).
This implementation does not reproduce their compiler/type-system guarantees.

The HMAC implementation uses the Python standard library and constant-time
digest comparison as recommended in the [Python documentation](https://docs.python.org/3/library/hmac.html).
The verifier holds the same key and could issue reviews itself: this is
shared-secret authentication, not public-key non-repudiation.

Concept-erasure research such as [LEACE](https://arxiv.org/abs/2306.03819)
establishes a particular linear guardedness property, not universal nonlinear
erasure of name meaning in this program. The architectural channel proof here
and the earlier empirical learned adapters answer different questions.
