# Audit findings and disposition

This revision audits the completed native-token activation experiment and
preserves its measurement records. Changes are isolated in this directory.

| Finding | Specific mechanism | Implemented response | Evidence and remaining scope |
|---|---|---|---|
| Backend failure allowed service reuse | A failed model call released the global lock without recording an invalid runtime | Poison the service on backend or model-guard failure, sanitize public error text and require a fresh service | CPU tests and actual observer fault injection exercise the failure and recovery paths |
| Close and concurrent calls lacked a service lifecycle | The global model lock alone did not encode per-service state | Add a nonblocking request lock, idempotent close, and explicit closed/poisoned checks | Tests cover same-service overlap, close while busy and global runtime contention |
| Invalid requests could be confused with model failure | Payload or native-boundary rejection is an input contract issue | Validate the authenticated role payload and layout before entering the model section | Bad requests remain recoverable and never invoke the backend in the instrumented test |
| The A/B rubric threshold favored acceptance | Raw qualification margins rose for better evidence, while the zero threshold misclassified negative cases | Fit a fixed scalar threshold on separate synthetic credential types and freeze it before held-out inference | The calibrated service is explicitly scoped to the credential task. Scores on arbitrary CV/job judgments continue to use the original readout |
| Review spans sometimes split a whitespace/name token | Native BPE can merge a leading space with name characters | Offer an unresolved, source-exact proposal absorbing only explicitly permitted adjacent whitespace | The reviewer must approve the expanded role. Punctuation or substantive evidence remains protected. Admission after a repair does not itself prove identical evidence across separately repaired names |
| A repair could conceal semantic reclassification | Expanding a span changes which source characters are considered evidence | Return the precise expansion, leave the proposal unresolved and let ordinary authentication sign the reviewed final payload | The source and full native IDs remain unchanged. Correct role semantics still require review |
| Token compatibility was insufficient to establish equivalent evidence | Separate whitespace repairs sometimes retained different token sequences | Bind interchangeable requests to an immutable approved sequence and reject exact-ID mismatches before inference | In the 120-case follow-up, 103 matched the anchor, nine additional token-compatible variants were rejected, and eight failed token compatibility |
| Digest equality is not an unconditional proof of sequence equality | A cryptographic digest has a finite range and therefore admits theoretical collisions | Compare approved token tuples directly; retain SHA-256 only as a reporting identifier | A test forces every digest to collide and verifies that changed evidence still rejects. The final contract passes the tokenizer sweep and GPU integration test |
| Optional repair could reject an already valid long formatting run | The repair search bounded whitespace before trying unchanged roles | The normal review preparation entry point tests the unchanged proposal first | Tests preserve valid 20-space formatting and still propose a needed one-space expansion. The frozen experiment helper remains preserved |
| Prior exact states might regress after lifecycle changes | Wrapping the inference entry point could alter execution or observation | Replay all 192 previous named cases through the normal public score method with internal research instrumentation | Compare every old score record and aligned stage digest, check original IDs and zero private slots |
| Clean model behavior might drift | A global readout calibration could affect unrelated prompts | Keep the credential profile an explicit operator configuration and retain an uncalibrated entry point | Replay the same 32 clean ARC-style vocabulary arrays against native inference |
| A fitted threshold artifact could be replaced accidentally | An altered file could change scores with the same named profile | Require its expected SHA-256 and exact schema when loading the published calibration | A changed threshold fails the artifact test. The trusted release manifest supplies the expected digest |
| The unit of experimental independence could be overstated | Names and prompt styles repeat the same few qualification conditions | Report credential types, condition groups and rows separately | Repeated names are conformance variants. The study supplies no population hiring-accuracy interval |
| Name discovery in free-form prose remains ambiguous | Identical strings can refer to an applicant, a client or a skill, depending on external context | Preserve the explicit reviewed-role input contract and reject incompatible requests | Additional prompt fitting cannot supply missing outside context or guarantee every unknown alias is found |
| Other racial proxies may remain in evidence | Job history, schools and policy choices can carry demographic information after name isolation | Keep the theorem limited to substitutions of designated identity fields with fixed evidence | Independent annotation, job-rubric labels and demographic outcome audits remain separate research requirements |
| Hardware-dependent dense attention can violate exact bytes | The prior real-CV diagnostic first diverged at fused attention under changing total lengths | Retain identical evidence shapes and values inside native blocks and restore original physical slots after each block | The new lifecycle retains the successful packed execution. The active computation schedule differs from original full dense attention |
| Universal model amnesia conflicts with retaining original named embeddings | The original named embedding lookup still distinguishes names before the boundary | State the protected boundary precisely and verify zero identity rows after it | The public scoring branch is isolated. Original tokens, pre-boundary embeddings, length and timing remain private observations |

## Why the semantic boundary cannot be removed by assertion

Consider the same text, “Jordan supplied the qualification evidence.” In one
external context Jordan is the applicant. In another Jordan is an independent
assessor. A deterministic role detector given only that identical text returns
the same roles in both worlds, so it must be wrong in at least one world if the
required roles differ. Additional trusted context or a restricted input format
is necessary to eliminate that ambiguity.

Similarly, an unchanged original named embedding stage cannot simultaneously
be identical across two names whose embeddings differ. The current guarantee
begins after the model-internal gate. Those premises are part of the property,
rather than empirical defects that more threshold fitting can remove.

## Research basis

Zhao et al. document sensitivity to answer-label priors and contextual calibration.
The present threshold is fitted with labeled synthetic credential examples,
which differs from their content-free calibration method.
[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)

Hugging Face documents tokenization and alignment. This code verifies exact
full-source token factorization directly and treats expanded role proposals as
review artifacts.
[Tokenization pipeline](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline)

The span-veto literature conditions exact isolation on policy, geometry and
execution assumptions. Our packed recurrence handles variable name length by
holding evidence computation geometry fixed.
[What the Model Cannot See](https://arxiv.org/abs/2610.15883)
