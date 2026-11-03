# Formal claims, including an impossibility boundary

## 1. What can be established universally

Write X=(N,Q), where N is the designated identity channel and Q is reviewed retained
evidence. For a deterministic fixed scoring network G, define S(N,Q)=G(Q). For
every admitted name value, including unseen names, S(N,Q)=S(N',Q). Every scoring
state H_l is similarly a function of Q. This is an exact conditional dependence
claim, not a statistical extrapolation from a list of names.

The native-evidence reference uses the same G(Q), so the complete evidence branch
matches its computation. This retains its errors as well as its capabilities.
It does not preserve the unrestricted named scorer G(N,Q), whose name dependence
is precisely what the intervention changes. Requiring both S(N,Q)=G(N,Q) for all
inputs and S(N,Q)=S(N',Q) is possible only if the original G was already invariant.

For two native named scores a and b, any common corrected score c obeys
max(|c-a|,|c-b|) >= |a-b|/2 by the triangle inequality. Thus exact name invariance
and exact preservation of two unequal original scores are incompatible. Aggregate
task accuracy can still be preserved; that is a different empirical claim.

## 2. What cannot be established with the current frozen raw branch

Suppose two admitted name substitutions yield distinct original embedding states
H_raw(N,Q) != H_raw(N',Q). If the source is embedded intact with unchanged weights,
this state remains part of execution. Therefore the proposition "every internal
state is independent of the name" is false. A later adapter or scoring gate does
not retroactively alter that earlier state. This argument needs no assertion
about racial identity or every possible decoder.

In the experiment, compare entire raw embedding arrays directly (including their
shape) across four names for each fixed Q. If all four are distinct, the mapping
from the four-element intervention set to H_raw is injective. Under a uniform
choice among those four names, H(N|H_raw,Q)=0 and I(N;H_raw|Q)=log2(4)=2 bits.
When all public scoring states coincide, I(N;H_public|Q)=0 for that same finite
intervention distribution. Neither quantity describes worldwide racial identity,
general decoding accuracy or every side channel.

Preserving a model's general knowledge of names is compatible with isolating its
hiring score from them. The publishable question is therefore **decision-path
noninterference**, not whole-model amnesia. Timing/admission/private diagnostics
remain excluded. The known inserted header role is exact by construction; all
original body names, historical discrimination and other proxies remain in Q.

## 3. Joint affine editing: a useful mathematical audit

Karvonen and Marks (2025), Section 3.4 [26], apply summed affine directional edits.
If U has unit columns and target projections b, the summed map is

    h_naive = h - U (U^T h - b).

Its residual is

    U^T h_naive - b = (I - U^T U)(U^T h - b).

Unless the directions are orthonormal (or this particular residual lies in the
required nullspace), one pass does not exactly impose all target projections.
This does not negate the paper's empirical fairness results: its method is not a
claim of exact joint equality. For full-column-rank U, the least Euclidean change
that satisfies U^T h'=b is obtained by constrained least squares:

    h_joint = h - U (U^T U)^(-1) (U^T h - b).

For rank-deficient U, use the Moore-Penrose pseudoinverse; exact feasibility
requires b in the range of U^T. The formula is standard linear algebra, not a
novel concept-erasure algorithm. It only controls chosen linear projections;
nonlinear information, distribution shifts and fair behavior are not certified.
The included rational finite example verifies the orthogonality caveat without
training or claiming to reproduce that paper's model experiments.
