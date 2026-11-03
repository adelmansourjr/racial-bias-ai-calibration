# NameGuard: controlled real-CV validation and novelty assessment

**Research release 0.4.0.** This separate study tests the frozen name-channel
scorer on real CV/job excerpts, compares it with equivalent external evidence
canonicalization, and checks whether private name information remains.

The result supports **exact decision-path name invariance under known roles**.
It does **not** support whole-model name/race erasure or an autonomous detector
of every identifying reference in arbitrary CVs.

## Completed experiment

| Measurement | Result |
|---|---|
| Real CVs / occupations / job contexts | 32 / 8 / 64 |
| Names per context / counterfactual rows | 4 / 256 |
| Actual MLX forwards including timing/warmups | 768 |
| Raw mean CV naming-cue probability range | 9.75 percentage points |
| Raw versus corrected decision-flip contexts | 9/64 versus 0/64 |
| Corrected matched-name score range | Exactly 0 |
| Direct counterfactual public-stage comparisons | 6,144, all equal |
| Native-evidence hidden/full-vocabulary comparisons | 128, all equal |
| Category-pair relevance diagnostic, raw / corrected | 62.5% / 68.75% |
| Paired relevance difference interval | -0.78 to +14.06 percentage points |
| Median paired internal/external latency ratio | 1.002 |
| Private raw embedding distinction | All four names distinct in every context |
| CPU tests | 60 pass |

The category metric is **not human hiring accuracy**. Its paired interval includes
no improvement; no formal utility noninferiority result is claimed. Native
evidence computation matches exactly on all tested contexts. The raw probability
range measures name sensitivity, not verified racial-group discrimination.

The strong external-Q baseline is approximately as fast as internal routing.
The previous 4.10x speedup applied to a scalar-encoding comparator and is not
presented as a speed advantage over ordinary optimized canonicalization.

## The central finding, simply

The model first embeds the complete named source. Reviewed identity roles then
control an evidence branch built inside the backend. That branch gives identical
scores when only the designated header name changes. The original model still
knows names, and its private raw states still distinguish the names.

For a uniform choice among the four tested names at fixed evidence, the private
state retains exactly 2 bits of that finite name variable; the identical public
scoring state carries 0 bits. This is a mathematical consequence of directly
observed pairwise distinct/equal states, not an estimate of general racial
information or open-world decoding. See [formal boundaries](docs/FORMAL-CLAIMS.md).

## Literature and novelty

The [expanded 31-source manuscript](paper/manuscript.md) includes in-text citations
and a [BibTeX bibliography](literature/references.bib). The
[novelty assessment](literature/NOVELTY-ASSESSMENT.md) identifies close work on
internal hiring-bias mitigation and exact marked-span noninterference that was
missing from the first review. Broad “first method” and new-unlearning claims
are rejected. This is an implementation/audit artifact with new local
measurements; a novel general erasure algorithm has not been established.

[Read the standalone paper](output/pdf/nameguard-real-cv-validation.pdf).

## Verify without model weights or private CVs

From this directory, with Python 3.10+:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/verify.py
python scripts/literature_tools.py
python scripts/verify_release.py
```

These checks replay recorded coverage, source integrity and analysis. They do
not independently repeat GPU inference. The study's 12 preparation/inference
source files were frozen before inference; later analysis, documentation and
tests are separately covered by the release manifest. The bootstrap method,
seed and stopping rules were specified in the original plan. There was no
training or outcome-based model selection. The plan is local, not external
preregistration.

## Reproduce inference

The exact checkpoint is identified in `audit/checkpoint-manifest.json`; weights
are not distributed. It is an MLX affine-four-bit Qwen2.5-1.5B derivative. This
study trains no LoRA or QLoRA adapter and never changes the base parameters.
Clef and other training jobs are left untouched.

Obtain the sources independently and place them at the parent project's
`research/data/Resume.csv` and `research/data/jobs-first-rows.json`. Source hashes
are in `audit/plan.json`. The exact cached 61-row job snapshot and converted
checkpoint are not bundled, so exact GPU reproduction requires obtaining those
matching artifacts; the public result verifier does not have that dependency.

To create a **fresh sibling study copy**, copy this folder, then remove only the
copy's `audit/plan.json`, `audit/case-index.json`, `local-private/contexts.json`
and generated files in its `results/` directory. Preserve source code and do not
edit a completed study. With `requirements-mlx.txt` in Apple Silicon Python:

```sh
python scripts/prepare.py
python scripts/run_gpu.py --checkpoint /path/to/exact/frozen-checkpoint
python scripts/analyze.py
python scripts/verify.py
```

The runner refuses to overwrite completed results. Preparation uses the first
140 CV words and 40 job-description words. It is an excerpt study, not a complete
document audit. All original body names/proxies remain unchanged. Only the
inserted header role is known by construction; the two credentials are machine
fixtures, not two independent human reviewers.

## Publication and privacy

The public artifact includes hashes, categories, artificial intervention names,
scores, native-reference checks and timing records. Actual CV/job excerpts and
secret keys are excluded. Hashes support integrity, not anonymization; public-source records may remain
linkable. Source access does not remove applicants' privacy
interests; no real applicant records should be uploaded in issues or commits.

This is a transparent preprint draft and controlled research artifact. It is not
a peer-reviewed publication or a consequential hiring system. External
replication, adjudicated free-form identity labels and domain-expert hiring
rubrics remain unperformed. The [independent-review packet](docs/INDEPENDENT-REVIEW.md)
defines those steps without inventing reviewer approval or participant results.

Suggested portfolio wording: **“Built and tested a frozen-LLM name-channel
isolation system, ran 768 Apple GPU forwards on real CV/job excerpts, verified
exact scoring-state invariance and native-evidence parity, and documented why
that does not imply whole-model information erasure.”**
