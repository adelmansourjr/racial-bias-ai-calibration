# Verified credential coverage recovery

This follow-up repairs the earlier 32/48 definitive-coverage result by
prioritizing the latest applicable authoritative record. The frozen model
uses the original protected scoring path and supplies the same score vector
to every proposal arm. The shared verifier controls definitive decisions.
See the [report](REPORT.md) and [frozen protocol](PROTOCOL.md) for the proof,
held-out design, quantitative results and trust assumptions.

From the repository root:

```sh
python3 -m unittest discover -s nameguard-coverage-study/tests -v
python3 nameguard-coverage-study/verify.py
```

The read-only verifier requires no model or GPU. To reproduce inference,
use the pinned Apple MLX environment and build the
[frozen Qwen checkpoint](../nameguard-peer-review/README.md). Run each split
into a fresh directory.

```sh
CHECKPOINT=/path/to/pinned/qwen-1.5b-4bit
python3 nameguard-coverage-study/run.py --checkpoint "$CHECKPOINT" --split dev --output /tmp/coverage-dev
python3 nameguard-coverage-study/run.py --checkpoint "$CHECKPOINT" --split test --output /tmp/coverage-test
python3 nameguard-coverage-study/measure.py /tmp/coverage-test
python3 nameguard-coverage-study/historical_replay.py
```

The study includes synthetic fixtures only. The deterministic complete-ledger
rule is a reference control and uses no model. The one-proposal policy relies
on authoritative and complete supplied metadata. It does not establish
performance on free-text qualification extraction or real hiring validity.
