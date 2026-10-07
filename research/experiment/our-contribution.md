# What we added to the fourth process

The comparison is with the supplied slides and photographed definitions. We have not inspected the full thesis, so this is not a claim that these techniques are absent from that thesis or new to the research literature.

| Supplied material | Our implemented contribution |
|---|---|
| Stage four observes shortlisting-rate disparity and activation cosine/norm. | A runnable claim ledger separates behavioral sensitivity, retained activation information, usefulness and coverage, each with pass/fail/insufficient evidence. |
| Behavioral improvement need not erase activation-space bias. | Matched-name interventions test scores and complete fixed-pool shortlist ID sets; paired distances and held-out linear/nonlinear probes independently test activation information. |
| Mean directions describe aggregate activation differences. | We verify complete decision interfaces, pointwise differences and three disjoint correction-fit/probe-train/probe-test sets, avoiding a probe fitted only on exactly erased calibration differences. |
| Mitigation is model dependent. | We actually run Qwen1.5B and the trained Clef9B joint head, retain projection and quantized LoRA failures, and compare them against fixed-family corrections. |
| The challenge asks for a roadmap/framework/proof of concept. | We implement complete final-state averaging, prove controlled name invariance at that interface and fixed-pool ranking/top-k invariance, verify the assumptions numerically, and supply scripts, raw measurements, gates and deployment requirements. |

## Our solution in one sentence

**HiringCheck enforces a controlled-name-invariant decision input and then independently audits output behavior, retained internal cue information and hiring usefulness.**

The same fixed set of four names produces the same complete averaged decision representation for a fixed CV and job. The original decision head receives that state and fixed metadata. Its score is unchanged by the received name; fixed name-independent IDs then make the ranking and selected ID set unchanged. Qwen uses its measured final 1536-vector A/B decision interface; Clef requires the whole final token sequence and its schema/option inputs.

This removes declared naming variation at the corrected state and output. It does not delete all protected information from the original model's activations or eliminate proxies in qualifications. It does not establish useful hiring decisions without valid suitability labels.

## The central evaluation question

**Does removing behavioral bias imply removing activation-space bias? No.** An internal state H=(q,c) can encode a cue c perfectly while its decision reads only q. We therefore test both independently. Actual Qwen projection also illustrates the danger: every binary shortlist decision becomes accept-all, but fresh corrected-state cue probes still achieve 68.75% linear / 62.5% nonlinear balanced accuracy, and fixed-pool rankings remain sensitive.

Our quantized-base LoRA demonstration used MLX affine 4-bit quantization, not canonical NF4 QLoRA. Presence fell about54%, but affected shortlist pools rose from2/6 to3/6 and nonlinear shortlist cue detection remained100%. This failed alternative is evidence that a lower activation mean metric cannot substitute for the full evaluation gate.

## Last-slide challenge answers

**How is fairness established?** The proof establishes declared name invariance at the complete corrected decision boundary and fixed-pool shortlist. Independent tests look for remaining cue information and poor usefulness. Universal racial fairness and whole-model erasure remain unproved.

**What are the risks and safeguards?** Qualification proxies, unequal access, biased criteria and unknown hiring usefulness can remain. Use independently reviewed rubrics and labels, broader audits, protected demographic-data handling, human oversight and accessible appeals.

**How were the datasets used?** Real CV qualifications and job descriptions formed unchanged candidate pools for controlled naming interventions. Occupation categories were sampling aids, not suitability labels. This is a short reference, not the main contribution.
