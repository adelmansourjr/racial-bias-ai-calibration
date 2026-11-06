# Exact name-channel isolation with original native token slots

The experiment asks whether the original named input can reach the model as
native tokens while the scoring activations become independent of a designated
identity span. The comparison distinguishes token IDs, slot positions,
position encoding, activation bytes, and the public score.

## Four distinct failure modes

1. **A score edit leaves hidden information.** A rank-one final-state edit can
   change one A/B margin while earlier layers and orthogonal components retain
   the name. In the preceding [subtractive study](../subtractive-lora-validation/README.md),
   the fitted correction reduced average variation while failing exact equality.
   Here the identity slot is set to zero before the first attention layer and
   stays zero at every stored block boundary.
2. **A tokenizer can mix roles in a single token.** If the native input contains
   an indivisible token spanning identity and evidence, that token cannot be
   classified as either role without losing one of its parts. The scorer checks
   that full-source tokenization equals the concatenation of separately tokenized
   maximal role runs. A failure rejects the request before model computation.
   The separate tokenizations serve as a boundary witness. Only IDs from the
   full original source enter the embedding lookup.
3. **Physical position leaks span length.** RoPE gives the attention logit a
   relative-position dependence. If a private span grows by one token, an
   evidence pair across the span can change its relative offset even with a
   perfect key mask. For a two-dimensional rotary plane and query and key
   both equal to (1,0), the dot product is cos((j−i)θ). Changing the offset can
   change that logit. We give evidence tokens their ordered evidence rank for
   the score path. Original physical slots remain in the stored sequence.
4. **Floating-point kernels can leak geometry.** In real arithmetic, a dense
   causal attention mask with evidence-only RoPE has the same evidence recurrence
   as compact computation. Hardware can choose a different reduction or
   quantized multiplication kernel as full sequence length changes. We measure
   that effect separately. For exact bytes the primary path passes identical
   evidence shapes and values to the pinned native blocks and scatters the
   results into the original slots after each block.

Zeroing identity embeddings alone is insufficient for exact isolation. If
their keys remain visible, even an identity value of zero can change the
softmax denominator. The primary path excludes the zeroed slots from the
native block's computation. The dense controls mask those keys in every block.

## Boundary and proof

Let the original full tokenizer output be t of length n. Let K be the
ordered evidence-slot indices and G be gathering at K. Let S scatter a
compact sequence back into the full layout, with +0 in every identity row.
The boundary state is H₀ = S(G(Emb(t))) after the named embeddings are evaluated.
Each native transformer block Bₗ receives compact evidence coordinates:

```text
Hₗ₊₁ = S(Bₗ(G(Hₗ)))
G(S(x)) = x
```

For two accepted requests with identical evidence-role text runs, the boundary
witness implies identical ordered evidence IDs. Hence their gathered H₀ are
equal. The pinned native blocks receive equal shapes, equal values and identical
causal and positional inputs. Induction gives equal gathered states and equal
readout, assuming deterministic execution. At every stored layer boundary the
identity rows are exactly +0. For equal native identity geometry the full
post-boundary arrays are also equal. With different identity lengths the
original full arrays can differ in shape and placement. Alignment at evidence
slots is the correct equivalence relation.

The native retained-ID reference runs the original Qwen model on the gathered
ID sequence. This reference is mathematically equivalent to the scoring
recurrence. It differs conceptually from tokenizing a name-free text string,
because removing text can enable fresh merges between the adjacent evidence
runs. We measure whether these references coincide on this cohort and do not
assume equivalence in general.

This is explicit runtime information-flow control over authenticated roles.
It keeps all original IDs and storage slots, then packs public rows inside each
block. The rotary coordinates used for scoring are evidence-relative. Original
named embeddings exist before the boundary and remain different. A process
observing original IDs, pre-boundary state, geometry, runtime, or rejected inputs
is outside the score contract. The base model retains its general name knowledge.
The scorer has no learned name detector, shared cache, or automatic role
discovery. Qualification text may contain unmarked proxies. Exact independence
of the designated name channel is a narrow property of these admitted requests.

## Experiment

The source, dependencies and cohort were hashed before inference. Six fixed
names cover Latin, Arabic and Chinese scripts, with token lengths one through
seven at the extremes and an equal-length one-token pair. They are string
interventions with no demographic labels. Six substitutions per context are
crossed with two job contexts per CV, for 192 planned variants. The cohort uses
16 real résumé excerpts across eight categories. Sampling excludes IDs and
normalized exact excerpts from the three named prior studies. Semantic near
duplicates and all earlier local CV studies were outside this exclusion check.

Native scores provide a name-sensitive comparator. On eight predetermined
contexts, dense physical RoPE and dense evidence-relative RoPE isolate the role
of position and kernel geometry. The packed path is checked on every context.
The original IDs and full stored tensor geometry are checked at runtime.
Every aligned array is compared to the first name variant across the boundary,
28 transformer blocks, final normalization and full vocabulary. We also
compare the entire final evidence state and full vocabulary to the native
retained-ID reference. A two-span synthetic control checks repeated applicant
identity. A boundary-merging request and an altered signed packet test rejection.

For behavioral utility we compare A/B decisions on 48 controlled credential
pairs, matched-category job scores, and top-three membership under independent
name substitutions from the finite pool. The synthetic credential statements
have deterministic labels. Category matching is a coarse relevance diagnostic.
A 32-item ARC-style clean-input regression checks full-vocabulary no-op behavior
through the gate on this runtime. Warm paired timings include signature and
boundary validation, exclude tracing and model loading, and use one context.
They describe this local service workload. This is a conformance and diagnostic
experiment without independent human hiring labels, externally registered
hypotheses, or a formal hiring-utility noninferiority margin.

## Research basis

- Mark Russinovich (2026) formalizes serving-time span veto under fixed span
  geometry and highlights tokenization, position and cache assumptions. Our
  method uses original-token slots with evidence-relative coordinates to cover
  varying span lengths in the scoring branch. [What the Model Cannot See](https://arxiv.org/html/2610.15883v1)
- Su et al. give the rotary relative-position formulation underlying the
  variable-length counterexample. [RoFormer](https://arxiv.org/abs/2104.09864)
- Sabelfeld and Myers distinguish end-to-end noninterference from local access
  checks. The public score and private host state require different contracts.
  [Language-Based Information-Flow Security](https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf)
- Karvonen and Marks measure affine activation mitigation under hiring
  counterfactuals. Their empirical reductions motivate comparison with learned
  edits, while the gate addresses an explicitly declared input dependency.
  [Robustly Improving LLM Fairness in Realistic Settings via Interpretability](https://arxiv.org/html/2506.10922v1)
- Controlled name substitutions and shortlist effects are motivated by
  correspondence-style retrieval and allocation studies. The Wilson and
  Caliskan version includes the authors' correction. This study uses neither
  paper's group-level effect estimates. [Corrected retrieval audit](https://arxiv.org/pdf/2407.20371v3) · [Small Changes, Large Consequences](https://arxiv.org/html/2501.04316v2)
- Chen and Xiao report demographic recovery from retained prose and sensitivity
  to evaluation design. That is a separate problem from isolating a known name
  span. [Beyond the Name](https://arxiv.org/abs/2609.16501)
- The native tokenization witness follows the distinction between normalization,
  pre-tokenization and subword tokenization. [Hugging Face tokenizer pipeline](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline)

## Alternatives and where each applies

A fixed-length substitution preserves the full physical layout across the
pair. The dense physical-position veto tests that case with an equal-token
name pair. It lets a scorer keep native physical positions under the fixed
geometry contract. Fixed-size padding could extend that representation to
different lengths, at the cost of inserting slots and changing the submitted
native token pattern. A segmentation policy could block all attention across a
private span while keeping physical RoPE, at the cost of dropping interactions
between evidence tokens on opposite sides. Re-tokenizing the retained string
could restore native evidence text merges, at the cost of changing the input
IDs used for scoring. The implemented gate retains the source IDs and evidence
order and resets only the scoring coordinate system. Evidence rows on opposite
sides can still attend causally.

The 120-case tokenizer stress test is a separate synthetic admission audit.
It varies Unicode spelling, internal spaces, accents, punctuation and boundary
layout. Its rate describes those fixtures, not expected coverage on a hiring
population. The boundary witness is deliberately conservative and refuses
a signed span when independent tokenization cannot account for the native
sequence exactly.

## Observed dense-route failure

On the first real CV/job context, the one-token and seven-token identity variants
have identical gathered boundary embeddings, normalized values, Q/K/V
projections, and logically rotated queries and keys. Their fused attention
outputs first differ by up to 0.00390625. This controlled diagnostic holds the
evidence values fixed while varying total geometry. The full vocabulary
difference grows to 0.43338 in maximum absolute logit error by the final head.
A short synthetic prompt has exact first-block equality. These observations
localize a geometry-dependent numerical effect to the attention call on this
runtime. They do not identify its internal kernel reduction algorithm or
establish the same behavior on every device.

The packed recurrence avoids this observed difference by presenting the native
block with the same evidence values and shape under each identity substitution.
It explicitly performs an internal gather and scatter. Every stored layer
boundary has the original length and token-slot layout. The active block
computation has the evidence length. Claims of preserving the internal dense
attention pattern or the original physical RoPE schedule would be incorrect.

## Human role correctness and general fairness

The acceptance property starts from verified role spans. Our injected header
span is known by construction and two random fixture credentials authenticate
that role decision. Two signatures do not establish two independent humans,
and lexical registry coverage cannot find every unknown alias or coreference.
Unknown identifying text retained as evidence can still affect the scorer.
Likewise a job policy or qualification proxy can encode racial disadvantage.
The result is exact isolation of the declared name channel after its activation
boundary. Broader racial fairness needs independent task labels, policy review,
and demographic evaluation with appropriate provenance.

The study observes only a constrained A/B score and the entire final-position
vocabulary vector. Equality of that vector supports the same next-token
distribution on the tested score prompts. It does not constitute an experiment
on arbitrary multi-token generated explanations. The runtime deliberately uses
fresh cache-free computation under its declared gate.
