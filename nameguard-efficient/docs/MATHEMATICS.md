# Native token efficiency without weakening the conditional name theorem

## 1. Diagnose the unnecessary cost

Let T be the original deterministic whole-string BPE tokenizer. For retained
Unicode evidence Q=c1...ck, the previous codec used

    C_scalar(Q) = T(c1) || ... || T(ck),
    m(Q) = sum_i |T(ci)|.

This guarantees invariance to annotation-field fragmentation, but blocks all
subword merges across scalar boundaries. Native token count is n(Q)=|T(Q)|.
The observed m/n ratio is not fixed across scripts or text: it must be measured.

Suppose a codec is required to be a concatenation homomorphism for EVERY
possible scalar segmentation: C(a+b)=C(a)||C(b). Induction forces the scalar
factorization and additive length above. Native BPE is generally not such a
homomorphism. Thus we cannot retain arbitrary independent-piece concatenation
and simply assert that all native cross-boundary merges remain available.

There are at least three different engineering options:

1. Keep fixed, trusted public field boundaries and use fieldwise BPE. This is
   cheap but weakens the equivalence relation: arbitrary resegmentation is no
   longer covered.
2. Keep scalarwise encoding. It is compositional and independent of annotation
   splits but has the measured token overhead.
3. Construct the canonical retained evidence as ONE sequence inside the model,
   then apply T once. This retains split invariance and native evidence tokens,
   at the cost of internal text processing and an extra original-source embed.

This project implements option 3, not a supposedly lossless posthoc projection
of context-dependent BPE tokens. General mixed source/evidence BPE tokens cannot
always be separated into native evidence token embeddings by merely dropping
source rows; the required token IDs and lengths may be different.

## 2. Canonical factorization theorem

Let D be a correctly reviewed typed original document. Let X(D) concatenate ALL
original node strings, and Q(D) concatenate only evidence-role strings, without
normalization or inserted separators. Both role order and punctuation matter.

The new model computes in this order:

    raw = Emb(T(X(D)))                    # complete original names included
    E = Q(D)                              # constructed INSIDE model backend
    H0 = Emb(T(E))                        # original native evidence embedding
    H_(l+1) = F_l(H_l; fixed theta)        # original positions/mask for E
    S(D) = R(H_L).

The raw branch cannot feed the canonical tokens, masks, states or readout.
Q(D)=Q(D') implies T(Q(D))=T(Q(D')), hence identical H0 and every subsequent
state by deterministic layer induction. Thus accepted identity changes,
new names and evidence-node resegmentation cannot change the score, PROVIDED
the reviewed role assignments preserve the same Q.

The equivalence no longer requires the tokenizer to be a homomorphism: we use
whole-string equality before tokenization. `audit/canonical-factorization.smt2`
checks congruence of arbitrary deterministic tokenizer/embedding/transformer/
readout functions under the supplied equal-evidence premise. It is a deliberately
small symbolic lemma, not verification that the program derives correct roles.

## 3. Exact token-count result, with honest accounting

Transformer length is EXACTLY n(Q), not approximately n(Q). It is the length of
the original tokenizer applied once to the same retained evidence. This is
native-token-count parity, not a theorem that greedy BPE uses the fewest possible
vocabulary tokens among all segmentations. No tokenizer optimality is claimed.

Original source names are still embedded before the canonical branch. Total
embedding token visits are n(X)+n(Q), and CPU tokenization/review validation adds
work. The raw named CV itself may be longer or shorter than retained evidence
under BPE boundary merges; we never assert n(X)>=n(Q) universally. The 1x token
claim applies to the SCORING TRANSFORMER relative to native retained evidence,
not total tokenizer calls, all embedding lookups or a native named-CV run.

For L layers, hidden width d and MLP width f, a rough prefill model is

    cost_transformer(n) = O(L*(n*d^2 + n*d*f + n^2*d)).

Replacing m with n reduces token-linear block work by m/n and the attention
quadratic term by (m/n)^2. Actual quantization, GQA, fused attention, overhead,
cache state and memory pressure determine wall time. These are scaling
comparisons, not promised end-to-end speedups. We measure warm paired service
latency, with signing and model loading excluded and scope stated.

## 4. Annotation correctness: what can be enforced

The input contains source-exact role spans, an explicit registered-identity
inventory, source-bound retained meanings for same-spelled skills/third parties,
a resolved flag and two credential endorsements of the SAME complete payload.
The authority rejects stale/overlapping spans, changed sources, known normalized
registry matches that remain unexplained, conflicting payloads, duplicated
reviewers, shared credential keys, unresolved reviews and signature tampering.

Two credentials do not mathematically prove two independent human judgments.
Agreement is not correctness: both may miss the same unknown alias. The
registered-name coverage checker only proves that its configured lexical and
normalization matches have been assigned roles. It does not discover every
transliteration, coreference or encoded identity channel.

Let A be the event that roles are complete/correct and all retained inputs are
fixed under a name intervention; let V be a public-output violation under a
model with a correct structural implementation. Then V is impossible on A:

    V subset A^c, therefore P(V) <= P(A^c).

If runtime/model correctness is an event B, the honest union bound is

    P(V) <= P(A^c union B^c) <= P(A^c) + P(B^c).

The probability model itself requires a defined deployment/input distribution.
This experiment estimates neither error probability from real CVs. If two
reviewers have error rates p1,p2, only under an additional justified independent
miss model would joint misses equal p1*p2. With correlated errors, joint misses
can be as high as min(p1,p2). We DO NOT assume independence or claim a squared
error rate from two signatures.

Character-level Cohen's kappa can describe agreement, not semantic truth:

    kappa = (p_observed - p_expected)/(1 - p_expected).

When p_expected=1 it is undefined, represented as null rather than 1. Character
observations within a CV are dependent; high overall agreement can hide missed
short name spans. Report identity-span/character recall and document-level
omission rates against independent adjudicated truth in a future human study.
The current oracle annotation experiment is synthetic and explicitly not such
a human study.

## 5. Performance and fairness limits

For fixed Q, native evidence hidden states and full FP32-input readout logits
match exactly, including native mistakes. Canonicalization is not fine-tuning
or semantic unlearning; it is role-based internal evidence construction after
the source enters the model. Names remain in the original model's knowledge and
raw embedding branch. Other proxies, job policy, unknown aliases, runtime timing,
rejections and population racial fairness are outside this conditional theorem.

Research sources: [tokenization pipeline](https://huggingface.co/docs/tokenizers/v0.20.3/en/pipeline),
[subword BPE](https://aclanthology.org/P16-1162/), and
[annotation agreement](https://aclanthology.org/J08-4004/). The construction and
conditional derivation here are our engineering analysis, not a claim that these
sources prove the full system correct or establish novel academic theory.
