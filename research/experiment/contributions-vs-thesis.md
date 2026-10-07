# What we added to the supplied thesis evaluation

This compares our implementation with the **supplied slides and photographs**, not the unseen complete thesis or the whole research literature. We make no claim of research priority.

## The starting point

Slide7’s fourth process evaluates shortlisting-rate gaps and post-mitigation bias directions using cosine and magnitude. The photographs additionally define normalized Presence, baseline-relative retention, balanced accuracy and layerwise differential. Slide8 already reports that behavioral improvement does not imply activation-space erasure, mitigation is model-dependent, and late-layer concentration does not identify a unique causal location. Those are the thesis’s findings, not our original contribution.

## Our implementation extends that evaluation in five ways

1. **From aggregate gaps to individual opportunities.** Matched name variants measure absolute score ranges and decision flips. Fixed job pools add rank displacement, effects on other applicants, cutoff crossings, pairwise reversals and complete top-k candidate-ID sets. A zero average gap can hide individuals exchanging opportunities.

2. **From diagnostics to an enforced decision interface.** The same fixed cue family produces one averaged complete decision state, h*=sum_a pi_a h_a, independently of the received name. Identical states imply identical head outputs; invariant scores plus opaque independent-ID ties imply invariant fixed-pool rankings. Qwen uses its measured constrained A/B state; Clef uses the aligned full sequence required by its trained head. This establishes declared cue invariance at that boundary, not erasure throughout the model.

3. **From one mean direction to paired and adversarial checks.** We measure individual paired state differences and linear/nonlinear cue probes, with separate correction fitting, probe training and evaluation applicants. Fresh names test a further generalization axis. This avoids chance probes trained only on calibration differences the projection was designed to annihilate.

4. **From a single fairness number to a claim ledger.** Behavioral sensitivity, encoded information, evidence of signal use, usefulness and coverage receive separate pass/fail/insufficient-evidence statuses. Corrected state equality cannot hide raw upstream information; low Presence cannot hide poor decisions. Real hiring usefulness stays unmeasured without validated suitability labels.

5. **From assumed success to retained failure evidence.** Actual Qwen projection achieved zero threshold flips while accepting every applicant; fresh probes still recovered its naming cues. Exploratory centering rejected every applicant. Actual MLX affine4-bit LoRA lowered Presence but worsened shortlist flips from0/16 to3/16; it was not NF4 QLoRA. Clef projection retained held-out cue sensitivity. These outcomes show why the extra gates matter.

Fixed-bank averages produced identical corrected states and naming-invariant fixed-pool selections on both actual models. Scores varied between applicants. Qualifications and upstream activations can still carry demographic information; neither nonconstant scores nor naming invariance establishes fair real-world hiring.

## Last-slide answers to the three challenge questions

- **Fairness:** Proves declared naming invariance at the final decision boundary; broader racial fairness remains unproved.
- **Risks:** Proxies, privacy and weak utility require reviewed rubrics, protected data handling, oversight and appeals.
- **Datasets:** Real resumes and job listings formed fixed pools; categories supported sampling, not qualification labels.
