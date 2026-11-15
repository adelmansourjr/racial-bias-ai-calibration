# Credential utility after exact name isolation

This follow-up diagnoses and tests a remedy for the accuracy decrease in the
previous name-noninterference study. Read the [experiment report](REPORT.md)
and [mathematical derivations](MATHEMATICS.md) for the full evidence and scope.

| Study | Result |
|---|---|
| Replay of the original synthetic cohort, post hoc | 168/192 old protected, 179/192 native, **180/192 new protected** |
| Fresh compositional test, 12 disjoint credential families | 41/96 native, 42/96 old protected, **69/96 protected with fitted readout** |
| Fresh wording test, 24 other families | Paired binary failed against native, **61/96 vs 73/96**. Cyclic three-way gave **83/96** on known cases and correctly referred **8/48** unresolved cases |
| Exact report and score checks | **32/32** historical conditions, **36/36** new wording conditions, **24/24** new compositional conditions |
| Reviewed structured-record policy | **1,024/1,024** scenario results correct, **4,096** byte-identical report comparisons |

The highest-accuracy readout on the compositional test incorrectly confirmed
**12/12 missing-evidence cases**. The independent structured policy checks
declared rules on reviewed observations, so its conformance results belong to a
different input contract. The [report](REPORT.md) records error counts,
motivating evidence, language and scope of contribution, and outstanding
human/domain validation.

The original named prompt reaches the tokenizer and embedding intact. The
existing gate removes the designated name's influence from protected
transformer activations under reviewed exact retained-token contracts.
Scores, logits and fixed-format explanation bytes are measured separately.
The base checkpoint is pinned and verified after inference. The ridge
readout was fitted only on 48 synthetic examples and remains separate from
the transformer weights.

## Reproduce the public records

From the repository root with Python 3.12:

```sh
python3 -m unittest discover -s nameguard-utility-study/tests -v
python3 nameguard-utility-study/verify.py
```

The verifier confirms the frozen sources, fixture coverage, paired summaries,
archival results and artifact manifest using the standard library. To rerun
the GPU studies, recreate the exact checkpoint using the documented
[Qwen reconstruction instructions](../nameguard-peer-review/README.md)
and install that study's pinned Apple MLX requirements.

```sh
CHECKPOINT=/path/to/pinned/qwen-1.5b-4bit
python3 nameguard-utility-study/experiment.py --checkpoint "$CHECKPOINT" --split dev --output /tmp/utility-dev
python3 nameguard-utility-study/experiment.py --checkpoint "$CHECKPOINT" --split test --output /tmp/utility-test
python3 nameguard-utility-study/analyze.py /tmp/utility-test
python3 nameguard-utility-study/compositional.py --checkpoint "$CHECKPOINT" --output /tmp/utility-composition
python3 nameguard-utility-study/analyze_compositional.py /tmp/utility-composition
python3 nameguard-utility-study/historical_replay.py --checkpoint "$CHECKPOINT" --output /tmp/utility-replay
python3 nameguard-utility-study/structured_policy_audit.py
```

Use new outputs for each GPU repeat. The scripts refuse to overwrite existing
plans and records. Bootstrap summaries treat credential families as the unit.
Hashes and synthetic inputs are public. GPU-dependent timings are descriptive.

Models and external data remain under their original terms. This package
contains the original synthetic cases, label definitions, hashes, and public
per-case outputs.
