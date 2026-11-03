# racial-bias-ai-calibration

## Latest: native-token-count NameGuard-Efficient

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
