# NameGuard evidence and risk study

[Research report](REPORT.md) · [Registered protocol](PROTOCOL.md) ·
[Follow-up protocol](RESERVE-PROTOCOL.md)

A frozen Qwen2.5-1.5B-Instruct model tests source evidence after its original-token
embedding boundary. Named input reaches the tokenizer and embedding lookup. The
reviewed name rows are excluded from subsequent scoring or generation, and the
public tokens use compact logical positions. Two experiments compare source
verification with and without first showing the model the proposed answer.
The public benchmark uses matched human-labelled SQuAD 2.0 cases on distinct
Wikipedia articles. This complements the earlier
[real-CV experiment](../nameguard-real-evidence/REPORT.md), which found that
traceable quotes can still support the wrong inference.

## Result

The candidate-aware verifier accepted 0 of 312 claims after failing its
development criterion. The candidate-blind follow-up confirmed 55 of 320 claims,
of which human labels marked 19 unsupported. The one-sided 95% finite-set audit
upper bound was 40%. Neither method passed the predeclared evidence gate.
The internal runtime passed both exact name panels. All 20 generation orbits
had identical output tokens, evidence report bytes and 375 matching direct
full-vocabulary comparisons across names.

## Reproduce the analysis on CPU

From the repository root, with Python 3.12:

```bash
python3 -m unittest discover -s nameguard-risk-study/tests -v

STUDY=nameguard-risk-study
REPLAY=$(mktemp -d)
curl --fail --location --silent --show-error \
  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
  -o "$REPLAY/dev-v2.0.json"
python3 "$STUDY/replay.py" \
  --source "$REPLAY/dev-v2.0.json" --output "$REPLAY/source"
python3 "$STUDY/analyze.py" \
  --development-labels "$REPLAY/source/queue/development-labels.json" \
  --holdout-labels "$REPLAY/source/queue/holdout-labels.json" \
  --output "$REPLAY/first-analysis"
python3 "$STUDY/reserve_analysis.py" \
  --queue "$REPLAY/source/reserve" --output "$REPLAY/reserve-analysis"
python3 "$STUDY/diagnostic.py" analyze \
  --queue "$REPLAY/source/reserve" --summary "$REPLAY/diagnostic-metrics.json"
diff "$STUDY/results/diagnostic-metrics.json" "$REPLAY/diagnostic-metrics.json"
diff "$STUDY/results/metrics.json" "$REPLAY/first-analysis/metrics.json"
diff "$STUDY/results/reserve-metrics.json" "$REPLAY/reserve-analysis/reserve-metrics.json"
```

The script checks the official dataset's SHA-256, source hashes, article split,
case IDs and labels. It reproduces the random audit draws from committed
permutations. Dataset text is downloaded separately. The official source is
published under [CC BY-SA 4.0](https://rajpurkar.github.io/SQuAD-explorer/).
The study release contains public question IDs, outputs, audit permutations and
hashes. It contains no model weights, original applicant records, or review keys.
The CPU run verifies the archived observations and statistics. Exact tensors
are checked in inference on the Metal run.

## Reproduce inference on Apple Silicon

Install [the pinned MLX requirements](../nameguard-peer-review/requirements-mlx.txt)
and [rebuild the original Qwen checkpoint](../nameguard-peer-review/README.md#reproduce-on-apple-mlx)
in a separate directory. The runtime checks each quantized checkpoint file and
the parameter digest. It leaves the weights read-only. From the same shell as above:

```bash
python3 "$STUDY/run.py" \
  --checkpoint "$CHECKPOINT" --plan "$STUDY/audit/plan.json" \
  --queue "$REPLAY/source/queue" --out "$REPLAY/first-inference"
python3 "$STUDY/reserve.py" run \
  --checkpoint "$CHECKPOINT" --plan "$STUDY/audit/reserve-plan.json" \
  --queue "$REPLAY/source/reserve" \
  --thresholds "$REPLAY/first-inference/thresholds.json" \
  --out "$REPLAY/reserve-inference"
```

Rerun each CPU analysis against the new observations with the corresponding
`--results` argument. Pass the reproduced first-run `--thresholds` path to
`reserve_analysis.py`. Run directories are fresh and never overwrite the
archived observations. For a genuine external replication, report the platform and
use a separately frozen dataset whose labels were inaccessible during development.

The final two controls use the frozen reserve case IDs. Use the archived
reserve artifacts as the diagnostic input to reproduce the recorded comparison:

```bash
python3 "$STUDY/diagnostic.py" run \
  --checkpoint "$CHECKPOINT" --queue "$REPLAY/source/reserve" \
  --out "$REPLAY/diagnostic-inference"
python3 "$STUDY/diagnostic.py" analyze \
  --queue "$REPLAY/source/reserve" --out "$REPLAY/diagnostic-inference" \
  --summary "$REPLAY/diagnostic-repeated.json"
```
