# Exact name isolation, credential utility, and explanations

## Property under test

Let \(X\) be the complete native-token CV prompt, including the applicant name.
Let \(E_j(X)\) be the ordered tokens retained by the reviewed identity policy
for prompt variant \(j\). An approved contract fixes this tuple before the
incoming name is evaluated. A request that cannot substantiate the same tuple
requires review. Token-witness feasibility and trustworthy identity annotation
are separate conditions.

The native embedding lookup receives all of \(X\). The activation intervention
zeros the designated rows and each block executes on the gathered evidence with
its own evidence-relative positions and causal mask. It scatters back to the
physical slots with zeroed identity rows. For a fixed deterministic runtime and
shared evidence tuple, the activation recurrence is

\[
A_{j,0} = \operatorname{Embed}(E_j),\qquad
A_{j,\ell+1}=B_{\ell}(A_{j,\ell}).
\]

By induction every aligned \(A_{j,\ell}\) is equal across admitted name
substitutions. The final observations include the full vocabulary array, not
just the returned decision. Original named embedding rows and sequence lengths
can vary. Equality of behavior alone would provide no proof about those earlier
representations.

Suppose a scoring method runs a fixed set of \(k\) prompts, each with its own tuple
contract. The method admits only when all \(k\) contracts are satisfied.
The proof applies separately to every admitted prompt variant. Their product
vector of readouts is equal, so any fixed deterministic composition of that
vector is equal. This includes a fitted readout whose trained parameters and
operation order are held constant. It also covers byte equality of a serialized
explanation if the explanation consumes only the invariant decision and
approved evidence, with fixed locale, formatting and tie rules. The implemented
language-model report renders the model's classification in fixed language and
does not assert an evidentiary quotation or a free-form rationale.

If admission itself varies with the name, the decision-or-rejection outcome can
still vary. More prompt variants can reduce coverage. For witness sets
\(V_j\), the admitted panel is \(\bigcap_j V_j\). Its coverage is bounded above by
the minimum single-variant coverage. On a uniform finite panel, the union bound
gives a lower bound of \(\max(0,1-\sum_j (1-|V_j|/n))\) when all applicable
assumptions use the same panel.

## Why preserved activations and preserved accuracy are separate

Deterministic name isolation constrains which predictors can be expressed. Let
\(x\mapsto e(x)\) be the retained evidence and \(y_i\) the independently defined
correct task label. In a finite evaluation table, partition cases by evidence
\(e\), and let \(n_{e,y}\) count the rows with evidence \(e\) and label \(y\). The
largest possible accuracy over all deterministic classifiers of the retained
evidence is exactly

\[
\frac{1}{N}\sum_e \max_y n_{e,y}.
\]

Proof. Every such classifier assigns one label to an entire evidence bucket and
therefore gets at most its largest class count correct. Assigning that
bucket's majority label attains the bound. Randomizing the class per bucket
takes a convex combination of its class counts and cannot exceed that maximum.
This is the standard empirical Bayes classification calculation applied to the
evidence interface.

If all labels match within each bucket, the benchmark admits a correct
invariant function. The proof supplies neither a training algorithm for
unseen evidence nor a claim the learned model finds that function. If removal
also deletes genuinely required evidence, conflicting labels can collide and
a score-only correction cannot resolve them. Conversely, native model score
differences by name do not automatically imply a task-accuracy ceiling. Prior
work in this repo proved a lower bound on matching name-sensitive **native
outputs**. Accuracy against independent evidence labels is a different
quantity.

## Prompt and readout hypotheses

For binary answer labels, write the two A-minus-B margins \(m_+\) and \(m_-\)
under swapped content assignments. Under the testable shared-offset model,
\(m_+=s+b\) and \(m_-=-s+b\). The antisymmetric part
\((m_+-m_-)/2=s\) recovers the semantic contrast and the symmetric part
\((m_++m_-)/2=b\) recovers the common offset. This algebra is exact under the
model. If the two prompts induce different content processing, their semantic
interactions remain in the result. A comparison with the same rewritten prompt
under one mapping is needed to measure what swapping actually contributes.
This is an evaluation arm because label sensitivity and calibration have
published precedents. [Zheng et al.](https://arxiv.org/abs/2309.03882)
and [Zhao et al.](https://proceedings.mlr.press/v139/zhao21c.html)

For three statuses, use three cyclic rotations. Let
\(z_{j,p}=s_{\pi_j(p)}+b_p+c_j+r_{j,p}\), where \(p\) is a letter position,
\(s\) is status signal, \(b\) is a shared position/letter effect, \(c\) is a
prompt-wide shift, and \(r\) is an interaction. Pull the logits back to semantic
order and average. Each status is seen at every position exactly once, so
\(\bar z_y=s_y+\bar b+\bar c+\bar r_y\). Differences cancel \(\bar b\) and
\(\bar c\). There is no guarantee that \(\bar r_y\) vanishes.

The learned arm uses a centered frozen final activation vector. Let \(X\) be the
row matrix of standardized training vectors after centering and scaling by one
positive global norm estimate, and \(Y\) the one-hot outcomes. With class prior
vector \(\bar y\), the ridge matrix is
\(W=X^T(XX^T+\lambda I)^{-1}(Y-\mathbf1\bar y^T)\) with \(\lambda=0.1\).
The matrix is positive definite for \(\lambda>0\). This is the ordinary
dual ridge solution for squared error with an unpenalized intercept after
centering. The research question is empirical: do the isolated states preserve
enough *decodable* credential signal to separate the statuses in new wording
and different credential families? Any positive result would be specific to
the measured readout and distribution. Readout weights are separate from the
frozen transformer. The readout fits no actual CVs.

## Deterministic rule when reviewed structured observations are available

The optional evidence evaluator accepts externally reviewed observations
bound to an opaque record ID, holder role, exact credential ID, effective date
and a valid/invalid finding. It filters to the applicant, exact credential
and dates up to the evaluation date. It then examines the latest applicable
observations with equal authority. Empty or conflicting latest evidence enters
review. Unanimous latest evidence determines the declared status. Sorted
winning record IDs produce a stable explanation identifying which evidence
controlled the policy result.

Given accurate observations and precedence assumptions, the evaluator computes
the declared rule exactly. Changing the raw name alone cannot alter this
function's output on the same observation set. This certificate concerns policy
execution, not whether the model can derive those observations accurately from
free text. It is a separate input contract so it cannot inflate the accuracy
reported for language-model inference. Real credential interpretation also
requires appropriate rules for revocation dates, jurisdiction, issuer authority,
and renewal, beyond the simple equal-precedence example.
