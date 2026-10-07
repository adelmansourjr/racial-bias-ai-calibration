# The solution: a cue-invariant decision boundary with separate evaluation gates

**We can remove the declared naming variation from both the final decision representation and its outputs. We have not erased racial information from every layer, the model’s weights, or the qualifications themselves.** Our actual Qwen and Clef experiments demonstrate that distinction. The central solution is an invariant decision boundary; the small LoRA experiment was a tested alternative that failed the joint evaluation.

## Does removing behavioral bias imply removing activation-space bias?

**No.** Let H=(q,c) encode qualifications q and a naming cue c, and let the decision read only q. Decisions are exactly cue-invariant, while the cue remains perfectly recoverable from H. Therefore behavioral invariance cannot imply activation-space erasure. Our evaluator asks both questions independently: do scores/ranks change under matched names, and do paired activation differences or independently trained probes still reveal the cue? In the actual Qwen projection pilot, zero binary shortlist flips concealed accept-all behavior and residual cue-probe accuracy of 68.75% linear / 62.5% nonlinear. Its rankings remained sensitive. A successful complete-state invariant construction is stronger than this misleading binary metric, but still does not erase every upstream representation or racial proxy.

## Five steps

**1. Hold the decision problem fixed.** Keep each applicant’s qualification record, job rubric, candidate pool and opaque application ID unchanged. Construct the same four-name family for every applicant, with fixed equal weights, independently of the name originally received. Qualification extraction must return the same record across the declared naming changes. This is the contract the evaluator checks; it cannot simply assume a generative extractor is unbiased.

**2. Compute every controlled variant.** Run the model on those fixed variants and capture the complete representation supplied to its hiring decision. For Qwen’s measured experiment this is the final state used by the constrained A/B readout. For Clef it is the aligned full sequence supplied to its trained joint head, together with its documented question/option inputs. Equal token lengths and alignment are verified where sequence averaging requires them. Raw branch states are retained for the audit.

**3. Enforce an invariant boundary.** Replace the received-name-dependent decision representation with

$$h^*(q,j)=\frac14\sum_{a\in\mathcal A}h(T_a(q),j).$$

The same qualification record creates the same complete family and the same average, whatever incoming naming cue is supplied. Feed only this averaged representation and fixed decision metadata to the head. Therefore the final representation is identical across the declared incoming cues, and any nonlinear head produces the same result. A separate fixed-family score mean offers behavioral invariance, but does not by itself establish representation equality.

**4. Rank the fixed pool with independent ties.** Sort the invariant applicant scores and break ties using opaque IDs assigned independently of names. The complete score vector, ranking and top-k selected IDs are then invariant under the declared name changes. The measured invariant scores remain different across applicants, so this result is not explained by globally constant scores. Useful hiring ranking remains a separate question because valid job-specific suitability labels were not supplied.

**5. Release a scoped evaluation result.** Report behavioral score/rank/top-k sensitivity; activation Presence with its numerator and mean-individual-norm denominator; paired state distances; independently trained linear/nonlinear probes; evidence of signal use; and usefulness/coverage. Use separate correction-fit, probe-train and final-evaluation applicants. Give each gate pass, fail or insufficient evidence. Neither zero mean differences nor chance performance of one probe is an erasure proof. Preserve upstream diagnostic results and failed methods.

## Does this remove both types of bias?

At the enforced boundary, it removes **the specified naming variation** from representations and decisions. Mathematically, if C is the controlled incoming cue and Z=h*(Q,J), then I(Z;C | Q,J)=0. In a randomized audit where C is independent of qualifications and job, I(Z;C)=0 as well. In actual populations, qualifications can correlate with race: I(Z;race) need not vanish. This conditional statement is not a claim that demographic facts, unequal opportunity or all proxies disappear.

Actual Qwen and Clef fixed-bank averages had zero within-applicant final-boundary naming variation and invariant fixed-pool selections. Their original branches retained cue information. Learned projections left residual held-out information and could distort decisions. The actual MLX affine4-bit, rank4 LoRA pilot reduced mean-based activation metrics but worsened shortlist naming flips and left evidence balanced accuracy at50%. It was not canonical NF4 QLoRA and did not succeed.

## Challenge-brief answers

**How is “unbiased” established?** By an exact, assumption-dependent interface proof for declared naming interventions, supported by actual measurements and separate uncertainty/utility gates. Universal racial fairness is not established.

**What are the risks and safeguards?** Names carry other social cues; qualification extraction, job criteria, labels and human reviewers can discriminate. Require reviewed rubrics, preserved qualifications, documented alignment, appeals and accessible oversight. Recheck after model, adapter, quantization or job changes. Small convenience samples cannot authorize deployment claims.

**How were the datasets used?** Actual resumes and job descriptions supplied fixed applicant–job inputs and pools. Occupation categories guided sampling; literal keyword labels supported a separate evidence diagnostic. Neither dataset supplied validated race identities or real hiring outcomes, so those quantities were not invented.
