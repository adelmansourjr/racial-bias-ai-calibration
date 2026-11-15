# Real CV evidence audit

[Report and preprint assessment](REPORT.md) · [Evaluation protocol](PROTOCOL.md)

Fresh inference uses real résumé excerpts and no trusted credential registry.
The prospective tests preserve protected scores and explanatory report bytes
across all 119 cases and four names per case. The automatic semantic checker
falls short of the joint zero unsupported confirmations and 60% recall target.

The question is whether the CV explicitly **reports using** a named tool in
work or a project. The reports link to exact excerpt offsets and a source
digest. Actual experience, employment validity and candidate suitability remain
outside the documentary task. Source texts, reference quotes, applicant-level
results and review keys remain private. The intervention strings and aggregates
are public.

## Checks and local reproduction

From the repository root:

```sh
python3 -m unittest discover -s nameguard-real-evidence/tests -v
python3 nameguard-real-evidence/verify.py
```

To make a new audit, put the independently downloaded dataset at
`research/data/Resume.csv`. Generate the cohorts with `sample.py`,
`followup_sample.py`, and `confirm_sample.py`. Prior private exclusion
files are optional for a new audit and influence the resulting cohort.
Use the labels protocol to record an exact positive quote or an empty
string for each case **before inference**. This study's private labels are
not bundled. Reproduce with a separately pinned Qwen checkpoint as described
in [the preceding runtime package](../nameguard-peer-review/README.md).

```sh
CHECKPOINT=/path/to/pinned/qwen-1.5b-4bit
python3 nameguard-real-evidence/gpu.py --checkpoint "$CHECKPOINT" \
  --cohort nameguard-real-evidence/local-private/confirm-cohort.json \
  --labels nameguard-real-evidence/local-private/confirm-labels.json \
  --verifier-module evidence_v3 \
  --output nameguard-real-evidence/local-private/confirm-run
python3 nameguard-real-evidence/analyze.py confirm
```

Use a fresh output location for inference. The summary script expects each
locally documented cohort, labels and run to share a private directory.
The locally archived run plans bind the exact dataset selections and
annotations. The public verifier checks source and artifact hashes and
aggregate accounting, and recomputes the private results when those files
are available. Passing public CI does not replicate the private evaluation.
