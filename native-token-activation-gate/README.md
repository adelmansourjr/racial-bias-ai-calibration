# Native-token activation gate for a frozen Qwen scorer

The model receives the complete named prompt with its original native token IDs.
A role-bound activation gate clears identity rows inside the model, computes the
evidence recurrence, and restores results to the original physical slots after
each block. Exact score and aligned-state equality passed all 32 real CV/job
contexts, with six names spanning one to seven native tokens.

This achieves a conditional name-channel isolation property. It requires correct
identity spans and a native tokenizer boundary check. The active transformer
computation gathers evidence rows inside each block. The stored sequence keeps
the original token slots and length. The scoring position coordinates follow
evidence order.

| Measurement | Native scorer | Activation gate |
|---|---:|---:|
| Exact scores across six names | 0/32 contexts | **32/32 contexts** |
| Exact aligned boundary, block and vocabulary arrays | 0/32 contexts | **32/32 contexts** |
| Name-related decision-flip contexts | 5/32 | **0/32** |
| Variable top-three job pools | 4/8 | **0/8** |
| Category-matched job wins | 10/16 CVs | 11/16 CVs |
| Qualification decisions correct | 49/96 | 48/96 |
| Correct direction under a qualification change | 48/48 pairs | **48/48 pairs** |

The qualification decision accuracy is poor for both scorers on this prompt.
The gate passes the exactness and directional tests, while useful hiring grading
remains unvalidated. Equal job-category diagnostics and a small clean-input
regression provide limited utility evidence.

## What is preserved and where the intervention happens

The input tokenizer runs on the entire original source. A second tokenization of
maximal annotated role runs checks native boundary compatibility. These auxiliary
calls never provide replacement input IDs. No original input token is changed,
inserted or removed. Every token receives its original embedding lookup.

At the model-internal boundary, identity rows become +0. Their original slots
remain present in stored hidden arrays. Every native Qwen block receives the
ordered evidence activations, with an evidence-only causal schedule and native
RoPE coordinates, then its result is scattered into the original array.

For two different-length names, the full arrays can have different shapes.
The proof compares matching evidence slots after alignment. For the equal-length
one-token name pair, a separate fresh process verified **248 full physical array
equalities** across eight contexts. Named raw embeddings differed in all eight
pairs and the post-boundary scores matched exactly.

This differs from the earlier internal text canonicalizer. The gate retains
original evidence token IDs instead of rebuilding and retokenizing evidence
text. In all 32 contexts, gathered retained IDs differed from directly
retokenizing the concatenated evidence text. The gate's native reference uses
the exact retained ID sequence.

[Proof, architectural tradeoffs and research citations](METHOD.md)

## Controls and actual measurements

The frozen main experiment executed **788 model forwards**. It sampled 16 CV
excerpts from eight categories and paired each with two job contexts. It
excluded IDs and normalized exact excerpts from three prior studies. Name
substitutions kept the CV and job evidence fixed. Names are string interventions
without verified race labels. A fit-free source hash and local cohort freeze
preceded inference.

- All 192 planned named inputs were accepted. Their identity lengths ranged from
  one to seven native tokens. All original IDs and stored slot geometry were
  checked during traced execution.
- There were **4,991 direct counterfactual array comparisons**, including the
  two-span control, and **64 direct native-reference array comparisons**. Every
  comparison matched. Identity rows were +0 in all 9,660 traced post-boundary
  stage checks across the packed and dense arms.
- The equal-token-length pair matched in both dense controls on all eight tested
  contexts. Across variable lengths, dense physical-position masking had mean
  logit range 0.385. Dense evidence-relative masking had mean range 0.217.
  The packed path had range zero on the full cohort.
- On a real CV, the first dense logical difference appeared at fused attention,
  after exact Q/K/V and logical RoPE values. The initial difference of 0.00390625
  grew through the layers. The short synthetic diagnostic matched in its first
  block. The measured localization supports identical compute shapes for exact
  bytes on this pinned runtime.
- The clean 32-item ARC-style regression had **identical full vocabulary logits**
  to the original model for every item, with 25/32 answers correct. The items come
  from an earlier local benchmark bank and use the project's choice readout.
- A synthetic tokenizer sweep tested 120 name/layout combinations. It accepted
  87 and rejected 33 at crossing role boundaries. Every accepted variant within
  a layout had the same evidence ID sequence. This is a conservative admission
  test over synthetic cases, not a population coverage estimate.
- A changed signed payload and a cross-boundary token request were rejected.
  Repeated identity spans produced the same gated output under substitution.
  The CLI reproduced the same unrounded margin for one-token and seven-token
  names in separate model processes.

The dense and packed warm service medians were 686 ms and 662 ms on one context.
The nine local pairs include a discarded initial warmup pair. Timing excludes
tracing and model loading. No general speed advantage follows from this small
local measurement. Model file hashes and the loaded parameter digest passed
final verification.

## Run the public checks

From the repository root, Python 3.10+ and its standard library suffice:

```sh
python3 -m unittest discover -s native-token-activation-gate/tests -v
python3 native-token-activation-gate/verify.py
```

The public package contains source, synthetic controls, aggregate results,
a pinned checkpoint manifest and the local design freeze. Actual CV/job text,
per-CV records, review secrets and original model weights are kept local.
The public verifier checks source and aggregate consistency. It reconstructs
private experiment rows when those local records are available.

GPU reproduction uses the environment pinned in
[requirements-mlx.txt](../nameguard-validation/requirements-mlx.txt). The supporting
Qwen loader and role authority come from `nameguard-validation/src/nameguard`.
Their source digests are recorded in `audit/plan.json`. Exact inference requires
the checkpoint matching `audit/checkpoint-manifest.json` and the recorded local
dataset snapshots.

## Score a reviewed named prompt

A newline-delimited name field avoids many tokenizer seam failures. Character
offsets refer to the exact UTF-8-decoded prompt. The local CLI assigns two fixture
credentials to the supplied roles. Operational semantic review remains a separate
human task.

```sh
python3 native-token-activation-gate/demo.py \
  --checkpoint /path/to/pinned/qwen \
  --prompt-file /path/to/reviewed-prompt.txt \
  --span 16:18
```

For a prompt beginning `Applicant name:\nAi\n`, the two-character name occupies
the interval 16:18. Multiple `--span` arguments support repeated mentions.
A role-boundary merge raises a contract error. The proof requires all identity
references targeted by the intervention to be correctly assigned. Unknown
aliases or other racial proxies retained in evidence can still influence scores.

## Scope of the exact result

The gate eliminates influence of admitted designated identity tokens on the
post-boundary evidence recurrence. The raw input IDs, initial named embedding,
sequence geometry and timing remain privately observable. The original model
also retains general name knowledge. This is explicit architectural routing
under reviewed roles, rather than an autonomously learned universal name eraser.

The closest prior span-veto work already studies exact internal noninterference
for fixed geometry. This implementation adds an inspectable native-ID boundary
witness, full original-slot storage and shape-stable evidence execution for the
tested variable lengths. Its ingredients follow existing information-flow
principles. The experiment establishes conformance on one pinned architecture
and runtime. Literature priority, independent human role accuracy, real hiring
validity, and every racial proxy remain open evaluation requirements.

See `results/summary.json`, `results/equal-geometry.json`,
`results/dense-audit-real.json`, `results/tokenizer-stress.json` and the method
document for exact measurements and assumptions.
