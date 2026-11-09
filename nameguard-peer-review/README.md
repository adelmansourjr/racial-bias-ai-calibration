# Admission-aware name noninterference: review package

This is a focused technical-report/preprint draft with runnable code, public
synthetic experiments, exact mathematical claims, two blinded AI reviewers,
recorded criticism and empirical follow-ups. It is prepared for external
technical review. Independent human validation and real hiring validity remain
open; this is not a completed PhD thesis or a racial-fairness certificate.

[Paper PDF](output/pdf/nameguard-admission-aware-preprint.pdf) ·
[Manuscript](paper/manuscript.md) · [Formal results](paper/THEORY.md) ·
[Literature and novelty](literature/review.md) ·
[Review responses](review-packet/RESPONSE-TO-REVIEW.md) ·
[Submission gates](SUBMISSION-CHECKLIST.md)

## What is new in this study

Equal scores among accepted CVs can hide name-dependent rejection. The new
compiler enumerates source-exact native-token boundary witnesses and chooses
one evidence contract with maximum support on a declared fitting-name panel.
It then checks disjoint test names against that same exact token tuple. The
mathematical optimum is finite and structural; correct semantic approval remains
necessary. The complete decision-and-score audit includes rejection.

Maximum-support enumeration and noninterference are established ideas. The
proposed contribution is their precise native-BPE feasibility formulation,
certificates and public comparative study. We do not claim the first internal
fairness intervention, a new universal erasure theorem, or superiority over an
external method supplied exactly the same evidence. The targeted literature
review identifies direct antecedents and does not certify field-wide priority.

## Main evidence

| Check | Result |
|---|---|
| Disjoint-name tokenizer panel | 12 fitting names, 24 test names, 8 layouts |
| Greedy contract / greedy witness | 158/192 structurally supported test cases |
| Greedy contract / exhaustive witnesses | 160/192 |
| Compiled contract / greedy witness | 154/192 |
| Compiled contract / exhaustive witnesses | **170/192** |
| Separate public synthetic GPU cohort | **192/192 admitted**, 32 conditions, 6 names |
| Exact same-evidence scores and aligned activations | **32/32 conditions** |
| Direct counterfactual/reference arrays | **10,912 exact comparisons** |
| Zeroed identity-slot stage checks | **5,760 passed** |
| Native name-dependent decision flips | 4/32 conditions; protected 0/32 |
| Equivalent external retained-ID method | All 192 stage comparisons exact |
| Official ARC-Challenge subset | **68/100 native = 68/100 protected**, full vocabulary exact |
| New synthetic credential accuracy | **179/192 native; 168/192 protected; 168/192 character deletion** |
| Balanced matched-service timing | About **1.1% internal overhead**, descriptive on this host |
| Checkpoint reconstruction | Every pinned file and parameter digest reproduced |
| Fresh-process full repeat | All 192 score/state records and 100 ARC records reproduced |
| Blinded AI annotation review | Two fresh contexts; 24/24 agreement, 6/24 uncertain |
| CPU mathematical and contract tests | 17 passed |

The new task accuracy loss is retained. The earlier 96/96 synthetic calibration
result concerns another task profile and is not reused to claim success here.
ARC checks the all-evidence path; it is not evidence that name intervention
preserves every hiring decision. These are constructed panels, not independent
population applicants. The timing comparison excludes compilation/human review
and was run while other approved training jobs remained active.

## Internal mechanism and exact claim

The unchanged original named source is tokenized and embedded intact. Within
the model execution path, reviewed identity rows are zeroed. Native transformer
blocks gather ordered evidence rows, use evidence-relative positions and scatter
back into the full original physical layout. Original names are not removed
before tokenization in this internal arm. Base weights are frozen and unchanged.
It is structural routing implemented in inference code, not successful QLoRA
forgetting learned into the base model.

With exact retained evidence e, the protected recurrence is

    A_0 = Embedding(e), A_(l+1) = B_l(A_l).

Equal e, fixed frozen parameters and deterministic state-free execution give
identical scoring states and outputs. An equivalent external retained-ID method
has the same property. General name knowledge and original raw embeddings remain
available outside the protected scoring branch. Admission, timing and private
sequence geometry can still depend on names.

The compiler proves maximum structural support under its bounded whitespace
policy on the fitting panel. It does not prove every future name is admitted or
that every proposal is semantically correct. It returns unresolved proposals;
only synthetic generator truth approves them in these tests. Public serving
continues to require authenticated source-bound role review.

## Reproduce without a model

From the repository root, Python 3.12:

    python3 -m unittest discover -s nameguard-peer-review/tests -v
    python3 nameguard-peer-review/verify.py
    python3 nameguard-peer-review/analyze.py
    python3 nameguard-peer-review/review_followups.py

The first two commands use only the standard library and sibling source
packages. They check mathematics, exact case coverage, frozen sources, full
31-stage metadata, benchmark counts and release integrity. They do not execute
GPU inference. The analysis commands deterministically reconstruct the archived
summaries and refuse inconsistent output replacement.

## Reproduce on Apple MLX

Use a separate Python environment and install requirements-mlx.txt. The measured
platform is recorded in results/gpu/runtime.json. Weights and benchmark source
material are acquired separately under their original terms.

1. Acquire the unquantized Qwen/Qwen2.5-1.5B-Instruct revision
   989aa7980e4cf806f80c7fef2b1adb7bc71aa306 from the official repository.
2. Rebuild into a fresh directory; the command never writes the source:

       python3 nameguard-peer-review/rebuild_checkpoint.py \
         --source /path/to/unquantized-source \
         --output /path/to/new-qwen-4bit \
         --report /path/to/new-reconstruction-report.json

3. Confirm matches_pinned_parameters and matches_every_pinned_checkpoint_file
   are true. The full inference loader rejects any checkpoint-file mismatch.
4. Download the immutable official ARC source. Every selected row is checked
   against the canonical content hash:

       python3 nameguard-peer-review/download_arc.py

5. Run the tokenizer and GPU studies into new output paths:

       python3 nameguard-peer-review/boundary_study.py \
         --checkpoint /path/to/new-qwen-4bit --output /tmp/new-boundary.json
       python3 nameguard-peer-review/gpu_study.py \
         --checkpoint /path/to/new-qwen-4bit --output-dir /tmp/new-gpu-study
       python3 nameguard-peer-review/analyze_run.py \
         --run /tmp/new-gpu-study --boundary /tmp/new-boundary.json \
         --output-dir /tmp/new-study-analysis

Completed study paths are never overwritten. The ARC download creates only local
private data; a clone initially has no such snapshot. Historical viewer-response
bytes can differ from the pinned-parquet JSON representation, but the question,
choice and answer rows are checked exactly. New run plans bind the downloaded
snapshot actually used. Timing and platform fields need not reproduce exactly.
The recorded fresh-process repeat used the reconstructed checkpoint on the same
host; no external laboratory replication is claimed.

## Review and correction record

Both annotation agents started with fresh contexts and instructions limiting
reads to the case packet and protocol. They saw no expected annotations, model
scores or implementation. They then performed separately labeled technical
reviews after their annotations were sealed. They found real problems, including
confounded comparators, timing pseudoreplication, an offset error, incomplete
cohort validation and semantic-support wording. Follow-up experiments and the
response ledger address them without rewriting the frozen primary study.

Agents share the provider/model family. Their agreement is not independent human
validation or evidence of real-world annotation accuracy. A blinded human review
packet, blank response form and analyzer are provided; no human responses were
fabricated. The protocol records its post-round uncertainty clarification.

## Artifact organization

- src/compiler.py: finite maximum-support contract selection and unresolved witnesses.
- src/fixtures.py: original public synthetic CV/job and name panels.
- src/comparators.py: matched-policy external retained-ID baseline.
- results/gpu/: prospective plan and original public synthetic per-case outputs.
- results/gpu-reconstructed/: fresh-process repeat on rebuilt weights.
- results/boundary-ablation.json: post-review 2x2 component comparison.
- results/balanced-timing.json: balanced paired timing follow-up.
- results/service-conformance.json: complete named stage metadata and timed-service bridge.
- audit/: frozen model manifest, data pins, failed/corrected reconstruction and follow-up plans.
- review-packet/: sealed AI annotations, original critiques, revalidations and human protocol.
- paper/: manuscript, derivations and PDF renderer.

Code and original synthetic material follow the included repository license.
Model and external dataset terms remain with their owners. No real applicant
records, CV-derived adapters, reviewer keys or model weights are released.

Before submission, the named author must verify and defend the derivations,
confirm authorship/contribution details and complete AI-assistance disclosures.
Independent human/domain review, broader model replication and field-wide novelty
assessment remain explicit external-review questions.
