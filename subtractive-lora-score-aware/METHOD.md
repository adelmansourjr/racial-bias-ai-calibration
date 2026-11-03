# Score-aware subtractive activation LoRA

The first SVD pilot is preserved in `../subtractive-lora-pilot-v2`. It selected a
rank-one centered projector on development criteria, retained 51/64 ARC-style
answers, but failed to remove held-out name sensitivity. New-name variation and
decision flips increased. The follow-up targets the score-relevant part of the
name contrast instead of assuming the largest variance direction is decisive.

## Fitting the cancellation, not reversing an arbitrary LoRA

The final normalized hidden vector is h in R^d. The original A/B margin is
m=w^T h, where w is the difference between the original A and B vocabulary rows.
For each fitting CV/job context, center the four name states around their mean.
Stack those centered states into D and the centered original margins into y.
Let M contain context-mean states centered around the global fitting mean mu.

Fit a cancellation vector v using

    min_v ||D v-y||^2 + lambda ||M v||^2 + rho ||v||^2.

The first term cancels paired name effects on the original margin. The second
penalizes changing mean scores across different evidence contexts. Ridge
regularization limits coefficient size. It does not force the correction to be
harmless on every possible qualification state. Context labels are not true
human hiring outcomes.

With T=[D; sqrt(lambda) M] and t=[y; 0], the solution is

    v = T^T (T T^T + rho I)^(-1) t.

The implementation scales rho by mean Gram diagonal. Candidate penalties,
strengths and selection rules are frozen before new confirmation inference.
The follow-up is adaptive exploratory research: its design was informed by the
first pilot and development-only algebraic diagnostics, not externally
preregistered. No outcomes from the sixteen newly selected confirmation CVs
were seen before adapter selection.

## A genuine internal rank-one residual adapter

Set a=w/(w^T w). Then w^T a=1 and apply

    h' = h - a [v^T(h-mu)].

The original head now yields

    w^T h' = w^T h - v^T(h-mu).

For the full vocabulary W,

    W h' = W h - (W a) v^T h + (W a) v^T mu.

Thus B=-(W a), A=v^T is a subtractive rank-one LoRA head update plus a
necessary affine offset. The implementation applies the residual to h before
the unchanged head, so it need not store a large vocabulary-sized B factor.
It is not merged into base weights. All original named input tokens pass through
the original transformer. No name recognition, annotation, anonymized alternate
or name-removal code operates on the input at inference.

The choice of a minimizes ||a|| subject to w^T a=1, by Cauchy-Schwarz. Therefore
for any fixed requested margin correction, it produces the smallest Euclidean
hidden-state change in that readout direction. This is not a theorem that all
other tasks or vocabulary logits are preserved. Earlier states remain unchanged,
and a rank-one correction generally leaves most name information recoverable.

## Exact success conditions

For a name-only state difference delta, the corrected margin difference is
(w-v)^T delta. It vanishes exactly if (w-v) is orthogonal to every such delta.
For a qualification-only difference g, the change in margin difference is
-v^T g. Exact retention requires v orthogonal to all relevant g. If name and
necessary qualification directions conflict, no linear cancellation can satisfy
both requirements. A finite fitting set supplies neither universal subspace.

An unrestricted signed LoRA already represents both additions and subtractions.
The structural objective and carefully measured direction supply the proposed
benefit, not merely negating its scale. Exact matrix identities may round
differently on FP32/quantized operations; actual edited-head errors are recorded.

## Literature and scope

This is a small application of established low-rank adaptation and regularized
least squares, not a new universal erasure theorem. Relevant primary sources:

1. Hu et al. (2022), *LoRA*, https://arxiv.org/abs/2106.09685v2 .
2. Belrose et al. (2023; manuscript v4, 2025), *LEACE*,
   https://arxiv.org/pdf/2306.03819v4 . Its stated linear-guardedness definition
   differs from the paired score-cancellation objective here.
3. Karvonen and Marks (2025), *Robustly Improving LLM Fairness in Realistic
   Settings via Interpretability*, Section 3.4,
   https://arxiv.org/html/2506.10922v1 . Their all-layer demographic intervention
   and fairness experiment are not reproduced by this final-state pilot.
