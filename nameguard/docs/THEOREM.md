# Segmentation-independent typed-identity noninterference

Let D be an accepted authenticated sequence of `(role, text)` nodes. Define
Q(D) as the **concatenated Unicode evidence text**, not a list of fields. Let
`t(c)` be the original tokenizer applied to one Unicode scalar, and define the
scalarwise codec by `C(s) = concat(t(c) for c in s)`. It follows directly that

```
C(a + b) = C(a) || C(b).
```

This is a different convention from ordinary whole-string BPE. The homomorphism
is deliberate: splitting or joining evidence nodes cannot change their retained
token sequence. Every original identity scalar is also encoded and embedded.

The internal gate collects all original token embeddings, then selects only
evidence coordinates in their original order. Pointwise embedding lookup gives

```
Z_0(D) = Emb(C(Q(D))).
```

Masks, positions and all 28 deterministic transformer blocks depend only on
this compact tensor. Readout labels and original frozen weights are fixed.
Induction on layers therefore yields, for any two accepted authenticated D,D':

```
Q(D) = Q(D')
  => Z_l(D) = Z_l(D') for every public layer l
  => margin(D) = margin(D')
  => PublicScore(D) = PublicScore(D').
```

The theorem quantifies over every accepted identity value, script, spelling,
length, number of identity fields and evidence segmentation. It makes no grading
accuracy assumption. It does require all identity information being varied to
be labeled identity. Unrecognized names left in evidence are not covered.

Authentication adds a separate **computational integrity premise**: without a
review key, an attacker cannot feasibly alter the signed source/roles/policy
under the security assumptions of HMAC-SHA256. It is not a semantic proof of
correct roles. If two validly reviewed documents have the same Q, they have the
same public score regardless of their different authentication tags.

The SMT files check abstract gather operations with arbitrary token arrays,
embedding functions, insertion locations and lengths. Single-block and two-block
checks return UNSAT for a differing retained activation. General finite-list
extension and transformer induction are written above, not machine-checked in
a proof assistant. Integer-valued symbolic embedding coordinates represent
arbitrary deterministic row values; they do not model floating-point rounding
or kernel implementations. Actual direct-array tests support conformance.

Two intentionally unsafe specifications return SAT: finite observed equality
can coexist with an unseen-name failure, and retaining source positions can
leak name length. Neither is silently discarded from the artifact record.

The theorem is **accepted-input, output noninterference**. It excludes whether
a request is admitted, resource exhaustion, runtime, logs, private research
traces and model-host compromise. It excludes other proxies and does not
claim a fair job policy, correct grading, whole-model name unlearning or causal
fairness with respect to actual race. Evidence correlated with names in a
population need not be unconditionally independent of names.

A constant scorer trivially satisfies unrestricted name invariance, but loses
all qualification dependence. This implementation instead remains nonconstant
across evidence. Universal automatic semantic-role discovery from arbitrary
ambiguous CV text has not been solved by this construction.
