# Name isolation with verifiable credential evidence

This study asks whether one frozen model can keep scores and evidence
explanations identical across applicant names while responding to changes
in the facts. It addresses an earlier scorer's false confirmations when
required evidence was missing.

[Study report](REPORT.md) · [Formal claims and literature](METHOD.md) ·
[Original protocol](PROTOCOL.md) · [Development-led test freeze](PROPOSAL-BUDGET.md) ·
[Exact authored prompt](witness.py)

The model reads the original name at the embedding stage. The inherited gate
zeros identity activations from its protected scoring path under an exact
retained-token contract. The new prompt asks the model to point to a registry
record. An independent checker evaluates the supplied ledger and returns a
deterministic explanation containing controlling record IDs or a reason for
review.

The checker provides correctness relative to the supplied authoritative
snapshot. The model determines which records are proposed within a budget.
A registry-only baseline computes that closed policy directly. No applicant
records or model weights are included.

## Reproduce or verify

From the repository root, with Python 3.12:

```sh
python3 -m unittest discover -s nameguard-witness-study/tests -v
python3 nameguard-witness-study/verify.py
```

The verifier recomputes metrics from every logged case, checks the exact
release hashes, and verifies that the numerical comparisons cover the
planned stages. To recompute the arrays, install the pinned MLX runtime
and recreate the [frozen Qwen checkpoint](../nameguard-peer-review/README.md).
Use new output paths for each run:

```sh
CHECKPOINT=/path/to/pinned/qwen-1.5b-4bit
python3 nameguard-witness-study/run.py --checkpoint "$CHECKPOINT" --split dev --output /tmp/witness-dev
python3 nameguard-witness-study/run.py --checkpoint "$CHECKPOINT" --split test --output /tmp/witness-test
python3 nameguard-witness-study/analyze.py /tmp/witness-test
python3 nameguard-witness-study/proposal_budget.py /tmp/witness-test
python3 nameguard-witness-study/opaque_replay.py --checkpoint "$CHECKPOINT" --output /tmp/witness-rekeyed
```

The model checkpoint and role-annotation dependencies are verified. Local
freezes document what was set before each test. The rekeyed run is an adaptive
design check. It uses an independent ID permutation and display order with the
same facts. It reports an exact expected uniform-ranking baseline, not a
demographic or population estimate.
