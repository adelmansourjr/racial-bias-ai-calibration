# racial-bias-ai-calibration

## Latest: hardened native-token gate and fixed credential readout

[Source, results and use](native-token-gate-hardening/README.md) · [Formal claim and protocol](native-token-gate-hardening/METHOD.md) · [Specific problems and fixes](native-token-gate-hardening/ISSUES.md)

The hardening revision preserves all 192 earlier CV scores, 5,952 aligned activation fingerprints and 32 clean full-vocabulary comparisons. It adds recoverable input rejection, poisoned-service handling after backend failures, concurrency/close checks, reviewed whitespace proposals and an exact immutable evidence-token contract. The original named input and native IDs reach the model intact; identity activations are gated internally and native blocks execute on shape-stable gathered evidence. Frozen base weights remain unchanged. This successful structural path is separate from the earlier learned LoRA pilots.

A fixed threshold fitted on separate synthetic credential types improves the held-out credential diagnostic from 48/96 to 96/96, with equal scores across names in all 24 condition groups and correct qualification direction in 48/48 pairs. This is a task-specific readout correction, not independently validated hiring accuracy. The original CV readout remains available with its prior exact scores.

The 120-case follow-up distinguishes 112 token-compatible variants from 103 variants matching the approved evidence anchor. Nine additional compatible variants are rejected for changed evidence and eight fail tokenizer compatibility. Direct token-tuple comparison avoids assuming hash collision freedom. GPU integration and separate CLI processes confirm exact short/long name scores. Twenty-nine CPU tests and public artifact verification pass.

The conditional guarantee requires correct reviewed roles and fixed evidence. Original raw embeddings and private token geometry still retain name information; general racial fairness, semantic name discovery and population hiring utility are not established. Source, synthetic examples and aggregates are public; real CV/job records, review keys, private states and model weights stay local. Earlier completed studies remain preserved below.

## Latest: original-token activation gating with variable name lengths

[Code and measured results](native-token-activation-gate/README.md) · [Mathematics, controls and primary literature](native-token-activation-gate/METHOD.md)

The frozen original-token scorer passed exact score and aligned activation equality in all 32 CV/job contexts across six names with one to seven native tokens. All 192 planned named inputs were accepted. Original source token IDs and full stored slot geometry were preserved. Inside each block the implementation gathers ordered evidence rows, uses evidence-relative RoPE and scatters back into the original slots. It has a different active computation schedule from ordinary full-length attention.

The main study ran 788 model forwards, with 4,991 direct counterfactual array comparisons and 64 native retained-ID reference comparisons. The ordinary scorer changed decisions in 5/32 contexts, versus 0/32 under the gate. The gate preserved all 48 qualification-change directions but had 48/96 binary rubric accuracy, versus 49/96 for the native scorer. Real hiring usefulness remains unvalidated. Thirty-two clean ARC-style inputs had identical full vocabulary arrays to the original model.

Dense logical masking still showed name-length-dependent numerical variation on this runtime. A diagnostic located its first difference at fused attention after exact evidence Q/K/V and logical RoPE values. The packed path resolves that observed geometry dependence. Five contract tests and fresh-process CLI checks pass.

The conditional result requires reviewed role spans and a conservative native-token boundary witness. The synthetic tokenizer stress sweep accepted 87/120 cases and rejected 33. General name knowledge, original raw embedding and private sequence geometry remain available to the host. The public guarantee covers the declared score path under the reviewed identity policy. The release contains source and aggregates, with private data and weights retained locally.



## Latest validation of subtractive LoRA and exact score equality

The held-out study used 24 CVs, 48 CV/job contexts and six new evaluation names. The learned subtractive adapter reduced the mean name-related logit range by 21.2%, with exact score equality in 0/48 contexts and decision flips in 2/48. It lost six correct synthetic qualification decisions. The role-bound control achieved exact score and final-state equality in 48/48 contexts and stable top-three membership in all eight job pools. Its guarantee uses the designated identity span and fixed evidence.

[Study, protocol, and aggregate results](subtractive-lora-validation/README.md)


## Latest: subtractive activation/LoRA mechanism pilots

[Score-aware source and aggregate results](subtractive-lora-score-aware/README.md) · [Mathematical derivation](subtractive-lora-score-aware/METHOD.md) · [Preserved SVD pilot](subtractive-lora-pilot-v2/README.md)

Two final-state subtractive adapters were fitted and evaluated on intact named prompts with frozen base weights. The score-aware follow-up uses sixteen additional CVs and 320 actual forwards; its rank-one activation subtraction is equivalent to a low-rank head update plus affine offset. Mean name-related probability range falls from 12.80 to 11.35 percentage points, but name-dependent flip contexts rise from 5 to 6 and a name-ID probe remains at 96.88%. The 64-item ARC-style regression changes from 51 to 52 correct; this is not a full benchmark or general capability noninferiority result. Category-pair relevance falls from 75% to 68.75% and is not human hiring accuracy.

**These variants do not solve universal name invariance or representation erasure.** Seventeen exact arithmetic tests pass, actual adapter execution is smoke-tested, and unchanged base parameter/file hashes are verified. There is no input name detector or redaction. This public release contains source code and aggregate summaries only: learned adapter binaries, per-CV rows, raw CV text and hidden-state caches are retained locally. Public CI checks mathematics and source integrity, not independent private-data reproduction. Earlier structural studies below remain separate and unchanged.

## Earlier: controlled real-CV validation and expanded novelty review

[Code, results and reproduction](nameguard-validation/README.md) · [Revised paper](nameguard-validation/paper/manuscript.md) · [PDF](nameguard-validation/output/pdf/nameguard-real-cv-validation.pdf) · [Novelty assessment](nameguard-validation/literature/NOVELTY-ASSESSMENT.md) · [Formal boundaries](nameguard-validation/docs/FORMAL-CLAIMS.md)

The v0.4.0 study ran 768 actual Apple GPU forwards on 32 real CV excerpts, eight occupational categories and 64 job contexts. Across four inserted names, the raw scorer's decision changed in 9/64 contexts; the role-isolated path changed in none. All 6,144 direct counterfactual scoring-stage comparisons and 128 native-evidence hidden/vocabulary comparisons were exactly equal. The mean raw CV naming-cue probability range was 9.75 percentage points. These are controlled name-sensitivity tests, not verified racial-group outcome estimates or human hiring accuracy.

The strong external canonicalization baseline is approximately equally fast (median paired internal/external ratio 1.002). Native-evidence computation is preserved. The category-relevance diagnostic was 62.5% raw versus 68.75% corrected, but the paired interval includes no improvement; this is not a formal hiring-utility noninferiority result.

Private raw embeddings still distinguish every tested name, while public scoring states coincide. The expanded 31-source review identifies direct antecedents for internal fairness editing and marked-span noninterference, so broad novelty and whole-model erasure claims are rejected. Sixty CPU tests pass. Real CV/job text, model weights and keys are excluded. External human validation and independent replication remain unperformed; the release includes a concrete review packet.

## Earlier: literature-reviewed NameGuard preprint

[Manuscript](nameguard-preprint/paper/manuscript.md) · [11-page PDF](nameguard-preprint/output/pdf/nameguard-literature-preprint-draft.pdf) · [Full literature review](nameguard-preprint/literature/review.md) · [BibTeX](nameguard-preprint/literature/references.bib) · [Code and reproduction](nameguard-preprint/README.md) · [Claim/evidence ledger](nameguard-preprint/docs/CLAIM-EVIDENCE.md)

The v0.3.0 release adds a targeted 25-source literature review with in-text citations, version/erratum checks, source locators and a search record. All 63 CPU tests pass (52 inherited plus 11 mathematical/citation checks). The original GPU measurements and frozen model source are unchanged; this revision adds no new model measurements or adapters.

The paper explicitly distinguishes score invariance from representation erasure. Its conditional guarantee comes from reviewed evidence routing, not a model learning to forget every name. An external canonicalizer providing identical evidence has the same scoring guarantee; raw embedding first adds no mathematical fairness advantage. The 4.10x timing result is against the prior scalar service, not optimized external anonymization. Real semantic annotation accuracy, hiring utility, independent replication and academic novelty remain unestablished. This is a transparent technical-report/preprint draft, not a peer-reviewed publication or deployment fairness certificate.

## Earlier: native-token-count NameGuard-Efficient

[Efficient code and results](nameguard-efficient/README.md) · [Revised preprint](nameguard-efficient/paper/manuscript.md) · [PDF](nameguard-efficient/output/pdf/nameguard-efficient-preprint-draft.pdf) · [Annotation protocol](nameguard-efficient/docs/ANNOTATION-PROTOCOL.md) · [Mathematical derivation](nameguard-efficient/docs/MATHEMATICS.md)

The efficient extension embeds the complete original named source, then constructs canonical reviewed evidence **inside the model** and uses ordinary BPE for the scoring transformer. It reaches exactly native-evidence token counts: 8,457 rather than 25,201 scalarwise tokens across 32 generated cases. The measured local median paired service speedup is 4.10x. All 108 name/segmentation conformance forwards preserve exact matched scores; 3,360 direct counterfactual stage comparisons and six native-reference array checks pass.

Annotation checks now require source-exact roles, declared-reference coverage and two credential endorsements of the same resolved payload. They rejected 108 deliberate registered omissions and 36 review conflicts. **This does not guarantee semantic role correctness or two independent human reviews:** an unregistered-alias negative control remains. The proof still assumes correct identity roles; total work also includes embedding the original source. Fifty-two CPU tests pass. The package is a preprint draft and reproducible portfolio artifact, not a peer-reviewed or deployment-ready fairness certificate.

The previous studies below remain intact; all new text/data are synthetic, with no model weights, real applicant records or secret keys uploaded.

## Earlier hardening study: NameGuard

[Code and reproducible tests](nameguard/README.md) · [Preprint draft](nameguard/paper/manuscript.md) · [PDF report](nameguard/output/pdf/nameguard-preprint-draft.pdf) · [Specific weaknesses and fixes](nameguard/docs/WEAKNESSES.md)

The new `nameguard/` package studies **authenticated typed-identity noninterference** inside a frozen Qwen scoring model. It adds source-bound review authentication, segmentation-independent input encoding, an internal activation gate, strict score-only outputs, runtime guards, solver specifications and 108 actual synthetic GPU forwards. The conditional proof covers designated identity fields; it does **not** prove universal name discovery or racial fairness in unannotated CVs. Thirty CPU tests run without downloading a model. Model weights, real applicant data and review keys are not included.

The package is a research prototype and preprint draft, with failed baselines and limitations preserved. It is not a peer-reviewed publication or a production hiring tool.

## Original datathon study

A reproducible implementation of the challenge's fourth process: **Evaluate**. It checks controlled naming-cue effects in model decisions and activations, tests corrections, and records the precise claims that pass, fail or lack evidence.

## Our solution and the central proof

**Fair outputs do not imply activation erasure.** The state `H=(q,c)` can encode a naming cue `c` perfectly while the score reads only the qualification component `q`. Our actual Qwen projection provides another warning: every threshold decision becomes accept-all while fresh cue probes remain above chance and pool rankings still change. Therefore behavioral and activation checks must be separate.

Our implemented correction averages a fixed name family at the **complete decision boundary**. Hold the qualifications `q`, job `j` and candidate IDs fixed; construct the same four cue variants independently of the received name; set `h*(q,j)=sum_a pi_a h(q,a,j)` with fixed weights. The same family gives the same complete state and fixed metadata. The original head therefore returns the same score. Name-independent tie breaking then gives the same ranking and top-k IDs. This removes received-header variation at that boundary; it does not erase cue information from earlier model branches or remove proxies in qualifications. Conditional on the fixed CV and job, the corrected state carries no information about the changed header.

For Clef, the boundary is the whole aligned final token sequence and its schema/option inputs; editing only a final token would miss inputs to the trained head. For Qwen, the measured boundary is the final vector used by a constrained A/B readout, not unrestricted generation. Projection and the small quantized LoRA alternative failed our combined checks; LoRA is not presented as the successful solution.

## Three challenge-brief answers

**How is the model's fairness established?** An exact interface argument establishes the declared naming-cue invariance, with numerical checks of score, rank, shortlist, alignment and decision metadata. Independent probes and utility/coverage gates report residual information and missing evidence. Universal racial fairness and whole-model erasure are not established.

**What are the sociotechnical implications and safeguards?** Stable decisions can still reflect unfair qualification proxies, biased requirements or reviewer labels. Actual hiring usefulness remains unvalidated. Broader audits, independent job-rubric labels, demographic privacy, reviewed human overrides and accessible appeals are required before deployment.

**How were both datasets used?** Real résumé excerpts and real job descriptions formed fixed applicant–job inputs and candidate pools. Occupation categories guided convenience sampling; literal keyword labels supplied a separate retrieval diagnostic. Neither source supplied verified race, suitability labels or linked hiring outcomes.

[What we added](research/experiment/our-contribution.md) compares our implemented evaluation framework with the supplied slides. The contribution is the working boundary correction, independent claim gates and reproducible experiment; it is not a claim of academic novelty or knowledge of the full thesis.

Using the same real CVs and job descriptions, changing names changed the top candidate in **one of six fixed pools for Qwen1.5B and one of six for Clef9B**. Fixed-family score and activation averaging made each ranking invariant across all 288 declared name assignments. This is a scoped algebraic guarantee with numerical checks. Actual hiring usefulness and universal racial fairness remain unestablished. Projection and adapter failures are retained.

## What is published

This repository publishes runnable evaluation code, mathematical explanations and the aggregate judging presentation. It excludes résumé-derived case excerpts, source records, individual scores/ranks, activations, adapter parameters and model weights. The complete local research package is retained separately. Automated publication review required removing applicant-level material before a public push.

The aggregate pilot findings are exploratory. Reproduction starts by independently downloading the public datasets and generating cases locally; the generated applicant material and model outputs remain local unless separately authorized for publication.

[Our contribution](research/experiment/our-contribution.md) · [Mathematical derivations](research/experiment/validated-math.md) · [Thesis comparison](research/experiment/contributions-vs-thesis.md) · [PowerPoint](presentation/hiring-bias-judges-pitch.pptx) · [PDF](presentation/hiring-bias-judges-pitch.pdf) · [60-second pitch](presentation/pitch-60-seconds.md)

## Reproduce locally

Python3.12; NumPy2.5.3 for analysis. Apple-silicon MLX/Metal for actual inference. The measured runtime used MLX0.32.3,mlx-lm0.31.3,Transformers5.18.0. No checkpoints are bundled.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlx.txt
python research/verify_fairness_math.py
python research/experiment/verify_photo_metrics.py
python research/experiment/download_data.py
python research/experiment/prepare_cases.py
QWEN_MODEL=/absolute/path/to/Qwen2.5-1.5B-Instruct
CLEF_MODEL=/absolute/path/to/clef-flash-mlx-8bit
python research/experiment/qwen_audit.py --model "$QWEN_MODEL"
python research/experiment/analyze_qwen_centered.py
python research/experiment/independent_qwen_validation.py
python research/experiment/clef_audit.py --model "$CLEF_MODEL" --inspect-only
python research/experiment/clef_audit.py --model "$CLEF_MODEL"
python research/experiment/independent_clef_probes.py
python research/experiment/qwen_qlora_demo.py --model "$QWEN_MODEL" --steps 24
python research/experiment/export_quantized_contrast.py --model "$QWEN_MODEL"
python research/experiment/independent_qlora_validation.py
python research/experiment/heldout_names.py --kind qwen --model "$QWEN_MODEL"
python research/experiment/heldout_names.py --kind clef --model "$CLEF_MODEL"
python research/experiment/ranking_audit.py
python research/experiment/ranking_audit.py --include-calibration --k 2 --output-dir research/experiment/ranking_secondary_full_pools
python research/experiment/cluster_statistics.py
python research/experiment/analyze_heldout_names.py
python research/experiment/evaluation_gate.py
python research/experiment/verify_artifacts.py
```

Run GPU models sequentially. Clef requires a compatible text-only8bit MLX conversion, the original trained `joint_head.safetensors`/config, and tokenizer files; the included support code preserves its full-sequence head. Qwen uses the original BF16 checkpoint and constrained A/B readout. The tiny adapter baseline is MLX affine4bit LoRA, not canonical NF4 QLoRA. Live dataset downloads may change; freshly generated cases need not equal the earlier local snapshot.

Optional scientific figures: `python -m pip install -r requirements-figures.txt`, then `python research/experiment/make_figures.py`. Generated cases, downloaded sources and all local outputs are ignored by Git.

Sources: [résumés](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), [job listings](https://huggingface.co/datasets/datastax/linkedin_job_listings), [name-audit research](https://www.nber.org/papers/w9873), [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Clef](https://huggingface.co/Cloudflare/clef-flash). Dataset/model terms remain with their owners; no blanket third-party license is asserted. Publishing this research code is not a hiring deployment.



