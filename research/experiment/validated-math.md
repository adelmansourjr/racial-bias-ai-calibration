# Validated mathematics for the stage-four hiring evaluator

This consolidates the slide challenge, photographed equations, actual-model failure cases, and fixed-pool ranking contract requested in the user attachment. It distinguishes exact algebra, measured finite-bank effects, and unsupported population claims. Existing actual Qwen/Clef/quantized-LoRA measurements are retained; no success is assumed.

## 1. Scope and challenge-brief answers

The challenge asks how a tool can check behavioral and activation-space bias, why its fairness claim is defensible, what risks remain, and how both datasets are used. The proposed answer is an enforced **evaluation gate** with separate behavioral, representation, usefulness, and coverage statuses.

The Kaggle resumes supply actual qualifications/excerpts; DataStax LinkedIn listings supply actual job contexts. Occupation Category is used for sampling, not as a job-specific qualification label. No linked hiring outcomes or validated race identities are available. Controlled name cues are experimental treatments; literal keyword labels are evidence controls, not overall suitability. Fixed pools preserve qualifications, job, rubric and opaque IDs under each assignment.

“Pass” means a declared cue-influence property on a documented input/decision interface and, where available, defensible utility. It does not mean all racial knowledge or structural discrimination has vanished. Retain failures, record uncertain quantities, and rerun the gate after quantization, training, job changes and distribution shifts.

## 2. Exact cue-invariant score and fixed-pool ranking

For qualifications q, job j, and fixed cue family A, define

$$s^*(q,j)=\sum_{a\in A}\pi_a s(T_a(q),j),\qquad\pi_a\ge0,\quad\sum_a\pi_a=1.$$

The same family, weights, qualification extraction and rendering must be used independently of the received name. Then s* contains no received-cue argument: s*(T_b(q),j)=s*(T_c(q),j). Arbitrarily nonlinear base scores are allowed. This removes declared cue variation, not an unmeasured proxy retained in q. Identical complete variants under deterministic inference give exact equality; stochastic scoring needs shared caching/randomness for pathwise equality or gives equality in distribution instead.

For a fixed pool I={1,...,n}, let independent scores s_i^* be assigned to opaque candidate IDs r_i that do not depend on names. Sort by the ordered pair (-s_i^*,r_i). Any allowed intervention leaves every s_i^* and r_i unchanged; therefore the entire sorted sequence and its first k IDs are unchanged. This proves:

- changed applicant rank is invariant;
- other applicants' ranks are invariant;
- every pairwise order is invariant;
- the complete top-k ID set is invariant.

Name swaps, one-at-a-time changes and balanced assignments are covered when they change only received cues and use the same invariant score rule. A name-sorted tie breaker violates the premise. A constant-score rule plus opaque IDs also satisfies this theorem while selecting arbitrarily; require a separate ranking-usefulness gate.

### Approximate ranking bounds

If every corrected score differs from an invariant reference score by at most eta uniformly over the declared interventions, a pairwise ordering is stable when its reference gap exceeds 2 eta. Top-k membership is stable when the reference kth versus (k+1)th gap exceeds 2 eta. If only one applicant changes and the others remain fixed, the relevant bound on that applicant's crossing of an unchanged score is its own allowed score displacement, not twice a common displacement. Measure empirical cutoff margins and actual membership reversals; do not infer stable ranking from a small average score gap.

These bounds are deterministic given valid uniform errors. A maximum observed test error is a finite-bank bound, not a uniform bound for every possible future resume or name.

### Joint-pool models need a separate theorem

Let a joint model emit S(P,a,o), where P is the full qualification pool, a the vector of cue assignments and o candidate order. The ranking theorem applies only if the **complete score vector** is invariant across every declared a, and the sorting IDs are independent. For order invariance, require the model's output to be permutation-equivariant after matching results to IDs, or explicitly correct order sensitivity.

A joint analogue can average S over the same complete family of cue-assignment vectors, independently of the received assignment, after matching each output to fixed IDs. This is algebraically invariant to the received assignment; a full Cartesian family can be costly. It requires utility checks and reliable structured ID matching. Averaging isolated scores for a model that later rereads the raw pool does not certify that contextual ranker. The present primary experiment independently ranks captured CV-job scores; a joint-pool guarantee must not be implied without an actual separately defined and tested interface.

## 3. Pointwise-difference and mean-preserving activation correction

Let h(q,a,j) be a complete decision bottleneck. Define the nuisance subspace

$$V=\operatorname{span}\{h(q,a,j)-h(q,b,j):\text{covered pairs}\},\qquad P=I-UU^\top,$$

where U is an orthonormal basis of V. For a calibration mean mu,

$$z=\mu+P(h-\mu).$$

Every covered difference satisfies z_a-z_b=P(h_a-h_b)=0. Therefore any downstream function consuming only z and fixed metadata is cue invariant. This permits nonlinear heads. An unseen difference outside V need not vanish; finite calibration fit gives no universal span coverage theorem.

For fixed P, the affine offset b minimizing E||Ph+b-h||² is b=(I-P)E[h]. Proof, with mu=E[h]:

$$E\|Ph+b-h\|^2=E\|(I-P)(h-\mu)\|^2+\|b-(I-P)\mu\|^2.$$

The cross term vanishes. Thus the mean-preserving correction is least-squares optimal **over constant offsets for that fixed P**. It is not necessarily the least-distorting map among all possible erasers. A calibration mean is an empirical optimum, not an established population expectation. Taskwise fixed means are valid when task identity does not change with the cue.

P can remove qualifications when signal directions overlap V; P=0 if V fills the whole state space. Centering changes absolute scores but cannot recover information deleted by P. In the actual Qwen pilot, the uncentered correction accepts every held-out applicant; exploratory centering rejects every applicant. Both demonstrate why low disparity cannot replace usefulness.

The complete-state condition is essential. Qwen last-token A/B readout genuinely uses that measured state for that constrained decision. It does not establish unrestricted generation invariance. Clef's trained joint head sees the full final sequence and question/option representations; a last-token-only correction leaves other paths open. Raw tokens, residual streams, masks, KV caches, question/option metadata and tools can also bypass a proposed boundary. Later computation preserves equality only when **all** its inputs are invariant.

For autoregressive answers, identical next-token transition kernels at every common history, including EOS, imply identical joint response laws by the product rule. One corrected next-token decision does not prove this for the full response.

## 4. Fixed-family activation averaging

For aligned complete states h_a and cue-independent weights, define h*=sum_a pi_a h_a and feed it to the same documented complete interface. Each received cue gives the same h*. Weighted squared-error decomposition shows

$$\sum_a\pi_a\|z-h_a\|^2=\sum_a\pi_a\|h^*-h_a\|^2+\|z-h^*\|^2.$$

h* uniquely minimizes distortion among constant replacements of that orbit. This is not a guarantee that an averaged state lies in the head's usual state distribution or preserves useful hiring rankings.

Clef full-sequence averaging requires verified token alignment, identical lengths/masks and stable question/option input. Unequal-length behavioral tests are legitimate, but averaging arrays without alignment violates this construction. A fixed bank of four names also does not automatically cover arbitrary new incoming names unless the incoming cue is independently canonicalized before constructing the same four variants. Evaluate new-name behavior of fitted projections separately; do not mistake an averaging interface's expected incoming-cue invariance for proof that the raw model is invariant on unseen names.

An orbit-mean output state can have zero measured cue difference while raw branch states still encode cues. Report both boundaries, not “all activation bias erased.”

Actual finite-bank checks show the invariant score outputs are **not constant across applicants**: Qwen orbit-activation shortlist scores range .16578–.50958 across the 16 held-out applicants; Clef orbit-activation shortlist scores range .01544–.76594, with 16 distinct values. Clef's invariant evidence scores range .02672–.95560. These ranges exclude a globally constant-score explanation for the invariant boundary. They do not establish that the differences correspond to legitimate suitability, which remains unmeasured without reviewed hiring labels. Qwen's orbit readout still rejects all evidence examples, so useful discrimination cannot be assumed from nonconstant probabilities alone.

## 5. Exact photographed metrics and their limits

The photographed definition is

$$v_\ell(M)=E[h_\ell^M(X)\mid A=a]-E[h_\ell^M(X)\mid A=b],\quad\operatorname{Presence}_\ell(M)=\frac{\|v_\ell(M)\|}{E\|h_\ell^M(X)\|}.$$

Use mean **individual norms**, not norm of the mean. For equally weighted two-group populations, Presence lies in [0,2]. Opposite deterministic states attain 2. Unequal group weights p and 1-p give bound 1/min(p,1-p). Zero denominator gives undefined 0/0, since the measured state is zero almost surely. Report numerator, denominator, pooling, layer and position separately.

A stronger paired diagnostic is

$$R_{\rm paired}=\frac{E\|H_a-H_b\|}{\tfrac12E(\|H_a\|+\|H_b\|)}.$$

For a finite positive denominator, Presence <= R_paired <=2 in a matched balanced design. R_paired=0 iff paired states are equal almost surely; empirical zero establishes equality on measured pairs only. Small positive values do not exclude a strong nonlinear decoder.

Equal group means can coexist with perfect nonlinear cue recovery: H|A=0 equally takes -1,+1 while H|A=1 takes -2,+2. The metric is zero; |H| exactly identifies A. Cosine with an old direction can vanish after rotation without erasure. Its zero-vector case is undefined. Background norm inflation can shrink normalized Presence without deleting information. Uniform scaling alone cannot shrink this normalized ratio.

The thesis retention is

$$R(M')=\frac{\operatorname{Presence}(M')-\operatorname{Presence}(B)}{\operatorname{Presence}(M_b)-\operatorname{Presence}(B)}.$$

It is unavailable without baseline B and injected M_b as separate measured conditions. It is undefined if the denominator is zero, unstable near zero and not generally bounded to [0,1]. R=0 means returned to baseline, not all bias gone.

BA=.5(TPR+TNR) requires defensible positive/negative job-qualification labels. Literal keyword BA is explicitly a diagnostic. A constant accept-all/reject-all rule has BA=.5 when both classes exist. Top-k always selects k even under constant scores, so check score variation and independently reviewed ranking quality as well.

## 6. Probe separation and actual findings

Use disjoint correction-fit, probe-train and probe-test applicants. Training a probe on calibration states whose cue differences were deliberately annihilated gives vacuous chance accuracy. The independent audits fixed linear ridge and nonlinear RBF kernel-ridge rules, with probe-train preprocessing and bandwidth only, before inspecting outputs. These are finite decoder families, not all possible attacks.

Fresh three-way probes showed residual empirical cue recovery in actual Qwen projected shortlist states (linear68.75%, RBF62.5%) and actual Clef projected summaries (pooled65.625%/59.375%, last65.625%/62.5%). This is consistent with nonzero held-out paired state differences. Eight probe-test applicants with four repeated names do not support broad population precision or significance claims. Same-name reuse tests applicant generalization, not name-family generalization; fresh names require their own measurements.

The actual 24-update quantized-base LoRA pilot trained only 19,456 last-block adapter parameters on the eight calibration applicants. It is MLX affine4bit group64 plus rank4 LoRA, not canonical NF4 QLoRA. Independently reconstructed before/after metrics exactly agree with the saved runner output using exported quantized A/B rows. Shortlist Presence approximately halves, yet name flips worsen0/16→3/16 and fresh protected-cue probes remain96.875%/100%. Evidence BA stays.5 while decisions change from all-B to all-A. Lower training loss or lower mean-direction magnitude therefore does not establish successful dual-bias mitigation.

Protected information being decodable is not itself proof of unfair causal use. Evidence of use needs controlled input/activation interventions and decision changes, with disruption-to-qualification controls. Low decodability also does not prove absence. Mean-based diagnostics, use evidence, output fairness and usefulness are separate gate entries.

## 7. Fixed-bank census and sampling uncertainty

When every declared name assignment of a given fixed pool is enumerated, the observed ranks and top-k changes are a **counterfactual census of that finite bank and pool**, subject to determinism and numerical measurement. No confidence interval is needed to describe the exact counted finite assignments. The census says nothing automatically about untested names, new applicants, new jobs or future deployment distributions.

For population inference, independent units are applicants for genuinely independently sampled applicant-level targets, or job pools for pool-level ranking targets. Multiple names, repeated assignments, order permutations and token/layer states are repeated measurements, not extra independent units. Interacting top-k outcomes share a pool; use pool-level event definitions. Six convenience-sampled pools support descriptive comparisons, not an assertion that the jobs are a representative iid hiring population.

If independent iid pools are assumed and no declared failures are observed in n pools, the exact one-sided confidence upper bound is

$$p_{\rm upper}=1-(\alpha/M)^{1/n},$$

for M fixed claims with union-bound multiplicity control. Six zero-failure pools give 39.30% at95% for one claim; 16 independent zero-failure applicants give17.07%. These numbers are conditional calculations, not justified population guarantees for convenience samples. The bound assumes the auditor/model/event definition is fixed and an honest untouched sample; exploratory choices after observed failures invalidate a prospective reading of the reused test set.

### Six-pool sign-flip diagnostics

For paired method differences d_j, j=1,...,6, an enumerated sign-flip distribution can use T=|mean_j d_j| and all2^6 sign vectors. An exact two-sided diagnostic is the fraction of sign vectors with |mean_j epsilon_j d_j|>=T, using inclusive ties. Its validity requires sign symmetry/exchangeability under the null for independent pool differences, or a genuine randomized assignment mechanism authorizing those flips. Paired observations alone do **not** justify randomly relabeling a trained correction as baseline.

Nonnegative harmful-change rates and their method differences are not automatically symmetric. Convenience selection of jobs does not establish exchangeability. If the assumptions are not defensible, label the computation an assumption-based sensitivity diagnostic rather than a calibrated p-value. For six nonzero same-sign differences the smallest possible two-sided sign-flip p is2/64=.03125; one-sided1/64=.015625 requires a predeclared direction. Zero differences, ties, multiple metrics, method selection and six-pool low power further limit interpretation. Prefer publishing every pool's effects and the finite-bank counts to claiming a broad significance result.

## 8. Sociotechnical safeguards attached to this scope

Names can signal gender, nationality, religion and socioeconomic background alongside racial association. Match surname/gender cues where feasible, disclose the name bank and token-length differences, and test new cues without changing qualifications. Do not infer real demographic labels from names. Separate non-discriminatory job evidence from proxies through a reviewed rubric rather than an undocumented extractor.

Job text, reviewers and historical labels can encode unequal opportunity. Audit rubric choices and reviewer disagreements; require privacy-aware, appropriately collected demographic/outcome data before real-population error-rate claims. Ranking redistributes a fixed number of opportunities, so publish entries/exits and full selected IDs, not just average gaps. Human review/appeals must be accessible and tested for unequal deferral burdens; an invariant automated score does not make a raw-name human decision invariant.

The defensible pitch is an evaluator that **detects hidden failures and enforces a narrowly proved interface**, then separately checks usefulness and residual information. Actual failed adapters and collapsed projection decisions are evidence that the evaluator catches false reassurance. Deployment remains unsupported without valid qualification labels, broader jobs/applicants/names, untouched confirmation and governance of the complete decision process.
