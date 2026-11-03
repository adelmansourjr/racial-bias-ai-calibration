# Reproduce without overstating independent validation

## CPU contract tests and saved-record replay

Python 3.10+ is sufficient, with no MLX or NumPy import for these commands:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/verify_results.py
python scripts/verify_manifest.py
```

These replay saved source hashes, coverage, scalar/native token counts,
timing summaries and annotation-case counts. They do not re-run GPU kernels or
reconstruct every omitted large tensor. The actual GPU experiment compared
arrays directly while they were in memory.

Z3 is an external executable used by `python scripts/check_proofs.py`.
Its UNSAT result checks a small conditional congruence lemma; SAT checks a
general independent-piece tokenization counterexample. This is not end-to-end
program verification. Solver versions can produce different proof/model text
while returning the same logical result.

## Pinned GPU runtime

Install `requirements-mlx.txt` on compatible Apple Silicon. The local study used
Python 3.12.8, MLX 0.32.3, mlx-lm 0.31.3, NumPy 2.5.3, Transformers 5.18.0,
Tokenizers 0.23.2 and Safetensors 0.8.0. These are recorded in `audit/environment.json`.
Use the actual frozen affine-four-bit Qwen2.5-1.5B runtime matching the supplied
file manifest and the parameter digest:

```
c6c622310d66489a5a9de5f438c23063bf784b3031f2e8af64bfd60da0394b40
```

The checkpoint is a local derivative of original Qwen2.5-1.5B-Instruct source
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, converted separately with
MLX affine quantization, group size 64. It is not an NF4 QLoRA checkpoint.
Base weights are not bundled. Other exports may fail the exact digest by design;
this intentionally limits portability and claims. The source model and tokenizer
remain subject to their original terms. Clef is not accessed.

The new package metadata is version 0.2.0; the preserved shared core module's
`__version__` remains its original 0.1.0 because the measured source freeze is
immutable. The canonical policy identifies itself as `nameguard-canonical-bpe-v2`.

To run a synthetic live demo:

```sh
python scripts/demo.py --checkpoint /path/to/pinned-qwen \
  --manifest data/tested-checkpoint-manifest.json
```

This uses two generated machine credentials and generator-defined roles. It is
not a demonstration of human review reliability. Requests in actual operation
must be reviewed under `docs/ANNOTATION-PROTOCOL.md`, with keys kept private.

For a new experiment, use a new directory or copy without completed `audit/design.json`
and result outputs. The runners deliberately reject overwriting frozen plans:

```sh
python scripts/experiment.py --checkpoint /path/to/pinned-qwen \
  --manifest data/tested-checkpoint-manifest.json
python scripts/annotation_study.py
python scripts/runtime_smoke.py --checkpoint /path/to/pinned-qwen \
  --manifest data/tested-checkpoint-manifest.json
```

The experiment freezes its source hashes before model inference. Later added
documentation, independent verifiers and release tools are recorded separately
in the final artifact manifest. No selected source is silently edited after
observing outcomes. Corpus/timing fixtures are completely synthetic.

## Timing interpretation

Each of three bodies has two warm-ups per method and seven paired repeats,
with method order shuffled using a fixed seed. The timer includes review
verification, source encoding, embedding, inference and per-call graph checks.
It excludes review signing, model load and full final parameter hashing.

There are 21 timing pairs (42 timed calls), plus 12 warm-ups, separate from
108 tensor-conformance forwards and three native-reference comparisons. The
canonical service verifies two credential signatures and registered coverage;
the scalar comparator uses the earlier single-signature API. This is a complete
service-package comparison, not an isolated ablation of token count alone.

Small local latency measurements do not establish hardware-independent speedups
or a population confidence interval. Token parity, by contrast, follows exactly
from using the same whole-evidence tokenizer and is checked on every corpus row.

## Manuscript

The preprint draft is `paper/manuscript.md`; its standalone PDF is rendered by
`scripts/build_pdf_v2.py` using `requirements-paper.txt`. The original copied
`build_pdf.py` remains unchanged because it was included in the measured source
freeze; the v2 renderer only changes this extension's output filename/title.
The package includes an
AI-assistance disclosure, limitations, negative controls, references and no
claim of external peer review, independent replication or academic novelty.
