# Subtractive activation/LoRA pilot: preserved negative and partial results

> **Public source-and-aggregate release.** Learned adapter binaries, per-CV score
> rows, raw text and hidden-state caches are retained locally and are not uploaded.
> Demo and full-record verifier commands require those separately held local
> artifacts. This public checkout supports the mathematical tests, aggregate
> inspection and source-hash verification, not independent replay of private rows.


Public checks (from the repository root):

```sh
python -m unittest discover -s subtractive-lora-pilot-v2/tests -v
python -m unittest discover -s subtractive-lora-score-aware/tests -v
python subtractive-lora-score-aware/scripts/verify_public.py
```

This controlled pilot tests final-state orthogonal name-direction subtraction
on intact named prompts. The base model and Clef remain untouched. It is not
the earlier name-role canonicalization method and uses no input-name detector,
redaction, anonymous alternative or identity metadata at inference.

Eight real-excerpt CVs fit an SVD name-contrast basis, eight CVs select rank,
strength and centering, and sixteen CVs evaluate the selected adapter. Four
additional test names are absent from this adapter's fitting/development. All
CVs came from an earlier experiment, so this is exploratory. The inherited
ordering couples fit/development CVs to plain prompts and test CVs to selective
prompts, so context transfer is confounded with the split. The new follow-up
confirmation includes both plain and selective contexts. The selected
rank-one centered full-strength adapter is a residual activation operator,
equivalent in exact arithmetic to a rank-one vocabulary-head update plus an
affine bias. No base matrix is modified or merged.

| Held-out measurement | Raw | Subtractive |
|---|---:|---:|
| Mean A/B logit range across eight names | 0.81339 | 0.78841 |
| Mean conditional A probability range | 0.13719 | 0.13375 |
| Contexts with name-dependent binary flips | 7/32 | 8/32 |
| Four-class name-ID probe accuracy | 98.44% | 71.09% |
| Category-pair relevance diagnostic | 81.25% | 81.25% |
| 64-item ARC-style regression accuracy | 79.69% | 79.69% |

On only the four additional names, mean probability range increased from
0.10150 to 0.10515. The adapter therefore **does not solve name invariance**.
Reducing recoverability of one name probe is not proof that decisions are fair;
that probe still exceeds 25% chance. Equal small-benchmark accuracy does not
establish general capability noninferiority; one prediction changed.

The pilot computes adapted score/capability metrics from cached states and the
compiled head formula. It validates direct activation-versus-factorized
full-vocabulary equivalence on eight held-out states, not on every pilot row.
The full-vocabulary factorization check has maximum FP32 error 8.58e-6. Exact
algebra is distinguished from numerical bitwise equality. Original base
parameter digest and checkpoint files verify unchanged after the run.

[Mathematical derivation](METHOD.md), `audit/plan.json`, `audit/selection.json`
and `results/final.json` contain the actual mechanism, selection and measurements.
The small `adapters/subtractive-basis.npz` file is a separately loaded overlay.
`scripts/demo.py` applies it after native inference to an unchanged prompt.

```sh
python -m unittest discover -s tests -v
python scripts/demo.py --checkpoint /path/to/exact/frozen-qwen \
  --prompt-file /path/to/named-cv-question.txt
```

The demo requires the original Apple Silicon MLX environment. Large original
weights, private CV/job text, hidden-state caches and a redundant compiled full
head factor are not distributed. The first attempt in the separate
`subtractive-lora-pilot` folder completed extraction but failed while serializing
a NumPy integer. This v2 run fixes that reporting error, saves its states and
preserves the original failed attempt. It is not concealed as a successful run.

The score-aware follow-up in `../subtractive-lora-score-aware` targets a specific
weakness exposed here: maximal name variance need not be the maximal causal
contribution to the A/B margin. Follow-up results are separate, with new CVs.

The released adapter is derived from real-excerpt activations. Excluding raw
text does not establish differential privacy or prevent all recovery of training
information; the learned file is not certified anonymized data.
