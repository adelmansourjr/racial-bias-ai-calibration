# Automated revalidation B

Reviewer: AI agent, reviewer B. Date: 2026-11-09.

This supplement assesses responses to B1–B9 in my sealed `technical-review-b.md`; it does not replace or edit that review or the earlier blinded annotations. This is automated artifact review, not independent human validation, independent model execution, or an external-laboratory replication. No other reviewer files were inspected.

## Scope and verification performed

I read the updated `paper/manuscript.md` and the requested follow-ups: `boundary_ablation.py`, `results/boundary-ablation.json`, `balanced_timing.py`, `audit/balanced-timing-plan.json`, `results/balanced-timing.json`, `service_conformance.py`, `results/service-conformance.json`, `review_followups.py`, `results/review-followups.json`, `compare_reconstruction.py`, `results/reconstruction-comparison.json`, and `audit/checkpoint-reconstruction-v2.json`. Large per-case JSON was inspected programmatically rather than reproducing every fingerprint in this report. I did not reread original raw results or imported backend code, run inference, inspect model weights, or inspect the reconstruction generator. Assertions about original-to-follow-up tensor equality are therefore supported by the submitted checking code and its recorded completion, not by my independent rerun.

I independently verified the following artifact consistency checks:

- Each recorded source hash matches the current boundary-ablation, balanced-timing, service-conformance, or review-follow-up source. The balanced-timing plan and result bind the same timing source.
- All factorial-ablation totals equal the sum of the eight layout rows.
- Balanced timing contains 80 distinct `(family, name, repeat)` tuples: eight families, two names, five repeats each. Recomputed within-repeat ratios, family medians, overall median, and mean match the published values. I did not independently regenerate the bootstrap percentiles.
- Service conformance contains 192 distinct `(context, name)` pairs covering a complete observed 32-by-6 cross. Every record lists exactly the ordered boundary, 28 block, final-hidden, and vocabulary stages. Published metadata has consistent expected ranks, float32 dtypes, hidden width 1536, and vocabulary shape `[1,1,151936]`. Finite tensor checks are implemented in the follow-up source; finite raw tensors cannot be independently established from hashes alone.

“Addressed” below means that the specific correction is adequate for the current narrow report; it does not confer broader scientific validation.

## Finding-by-finding status

| Finding | Status | Assessment |
|---|---|---|
| B1 Contract/witness confounding | Addressed | The factorial ablation separates the two components, and the manuscript reports the interaction honestly. |
| B2 Dependent timing case units | Addressed for the follow-up | The new endpoint aggregates paired repeats within eight families; the original dependent-case interval is retained and excluded from the efficiency assessment. |
| B3 Timing pairing, load and sessions | Partial | Pairing is repaired; multi-session/load generalization remains untested. |
| B4 Correctness/timed-service bridge | Addressed | All 192 normal service outputs are checked against each other and the original reference margins. |
| B5 Omitted deletion utility | Addressed | Deletion accuracy, prediction disagreements, and a family-paired comparison are added as post hoc results. |
| B6 Trace-completeness robustness | Addressed for the current cohort | Explicit ordered stage names, expected shapes, finite checks, and original-fingerprint checks close the silent-prefix gap empirically. |
| B7 Finite-panel generalization | Remaining scientific requirement | Claims are appropriately finite; no new representative name/layout/text sampling study has been supplied. |
| B8 Freeze/reconstruction provenance | Partial | Reconstruction evidence is substantially stronger; original analysis-freeze and independent replication limitations remain. |
| B9 Real task/human validation | Remaining scientific requirement | All-evidence ARC and AI annotation agreement remain narrow controls; the manuscript now states their limits clearly. |

### B1 — addressed

`boundary_ablation.py:25–31` implements all four contract-by-witness combinations. Published totals are 158, 160, 154, and 170 for anchor/first, anchor/all, compiled/first, and compiled/all respectively. Thus the comparison holding exhaustive witness matching fixed is **160 to 170**, a ten-case gain (5.21 percentage points) on this finite panel. The original 158-to-170 combined gain is not solely contract selection. The compiled contract is worse with first-witness matching (154 versus 158), making the interaction material.

Manuscript section 5.1 explicitly explains these facts and labels the ablation post-review. This resolves the attribution concern. The ablation still uses the same constructed names/layouts; it is a component explanation, not fresh generalization evidence.

### B2 — addressed for the new timing assessment

`balanced_timing.py:37–55` selects one direct positive context per family, uses two fixed names, obtains five shuffled pairs per name, and computes one median paired ratio per family. The result contains all 80 planned pairs and eight family summaries. Manuscript section 5.4 reports median family ratio **1.0111841**, mean **1.0117933**, and a descriptive family-bootstrap interval **[1.0110424,1.0126580]**. It explicitly says the earlier dependent-case result is not used for the efficiency assessment.

This repairs the original counting problem for the stated follow-up estimand. The eight families are constructed rather than sampled independent deployments; they do not supply eight independent host sessions. The interval should remain descriptive, as written.

### B3 — partial

The new family endpoint is based on each repeat's `internal_seconds/external_seconds`, so it retains the pairing discarded by the original analysis. Shuffle order is recoverable from dictionary insertion order in the raw timing JSON, although an explicit `order` field would be more portable. The preceding equality call also exercises both services before measurements.

The follow-up still appears to be a single host session with fixed context/name-block order, only positive/direct contexts, and concurrent training load. There are no independent sessions or quantitative load/thermal records in the supplied follow-up. A narrow description of approximately 1.1% online overhead on this measured host/session is warranted; a stable hardware-independent overhead or a population latency confidence claim is not. The updated manuscript respects this distinction. Multiple balanced sessions and explicit environment/load telemetry remain necessary for stronger performance claims.

### B4 — addressed

`service_conformance.py:56–60` compares the normal internal and actual timed external service result objects, checks their margins against both original compiled and original direct same-ID reference margins, and requires the internal trace fingerprints to match the originals. The result records 192 successful service comparisons. The timing code independently checks service result equality before each timed case (`balanced_timing.py:45`).

This closes the missing correctness-to-timed-service bridge for the observed inputs. It does not independently establish that the imported service enforces every stated authentication/lock/guard policy; I did not inspect that implementation. No such policy defect is demonstrated here.

### B5 — addressed

`review_followups.py:17–25` calculates deletion accuracy and family-paired changes from existing predictions. The post hoc result reports native **179/192**, compiled **168/192**, and deletion **168/192**, with 12 deletion-versus-compiled and 25 deletion-versus-native prediction changes. The deletion-minus-native family-bootstrap mean is -5.73 percentage points, with descriptive interval approximately [-13.54,+1.04] points.

Manuscript section 5.3 now states the equal aggregate accuracy and differing decisions, and correctly declines to derive a practical utility advantage from score inequality. The requested practical comparison is therefore present. Equal aggregate accuracy does not establish casewise equivalence or noninferiority; the manuscript makes neither claim.

### B6 — addressed for current-cohort evidence

`service_conformance.py:31,43` requires an exact model-derived ordered stage sequence. Lines 39 and 62–64 require finite arrays and expected shapes; lines 59–60 bind the trace to the original fingerprint sequence, including its full length. Published records contain 31 stages in every one of 192 unique context/name pairs. This excludes the original silent-short-prefix explanation for the current recorded traces.

The frozen original code still uses `zip`, appropriately preserved as part of provenance. The new program should additionally assert planned unique context/name coverage itself, rather than rely on loop construction and post hoc inspection. My artifact check verified complete observed coverage in this result. This is a future robustness recommendation, not an unresolved current missing-stage result.

### B7 — remaining scientific requirement, with adequate claim scoping

The new factorial experiment reuses the original names, layouts, tokenizer and fixed evidence construction. GPU conformance still supplies no new admission-gain cohort. It does not establish population naming-convention coverage, robustness to realistic CV text, multiple applicant mentions, role ambiguity, or demographic admission parity.

The manuscript accurately labels the gain finite, says layouts differ, denies demographic-parity conclusions, and retains the constructed-panel limitation. No additional repair is necessary for a carefully scoped finite feasibility report. Representative sampling and a predeclared repeated-measures generalization study are still required before making broader coverage claims.

### B8 — partial, with a materially improved reconstruction record

The new audit declares a specific source model revision, the same parameter digest as the original study, and exact recreated hashes for eight checkpoint/tokenizer/configuration files. `compare_reconstruction.py:9–22` requires equality on 192 rows across the specified substantive fields and fingerprints and exact equality of the 100 ARC records; the submitted result reports success. The updated manuscript distinguishes same-host fresh-process reconstruction from independent-laboratory replication and transparently describes the failed initial reconstruction check.

These artifacts provide stronger same-host reconstruction/repeat evidence. I verified the comparison logic and recorded result, not the reconstruction process or source-download recipe itself. The original local plan still cannot retroactively freeze an analysis that it did not hash, become externally timestamped preregistration, or prove absence of prior exposure. New follow-up labels appropriately preserve that distinction. Independent reconstruction from the public recipe, a complete immutable environment/analysis/data manifest, and independent execution remain desirable reproducibility requirements.

### B9 — remaining scientific requirement, with adequate disclosure

The updated manuscript continues to identify ARC as all-evidence preservation and reports the credential utility decline. Its annotation paragraph explicitly identifies the two reviewers as AI agents and their agreement as automated synthetic consistency evidence. It retains independent human role reliability, credible real job-domain labels, another model family, and external-laboratory replication as outstanding requirements.

I did not inspect the other reviewer's files or independently validate the reported inter-reviewer agreement. My own submission remains an AI annotation review. The new wording does not fabricate human validation, but the missing human/real-task evidence has not been supplied. Those requirements matter for a hiring claim; they need not prevent publication of an accurately bounded synthetic methods report.

## Overall revalidation judgment

The follow-ups substantively repair the main component-attribution, timing-unit, practical-baseline and exact-trace concerns. They are labeled adaptive/post hoc and do not overwrite the original studies. The updated report supports a finite tokenizer-feasibility and deterministic conformance contribution, with a measured utility loss and a small service overhead in one host session.

Residual requirements are scientific generalization and validation, not a request for more same-data conformance counts: realistic role/task annotation by humans, a broader prespecified cohort, independent reconstruction/execution, additional models/runtimes, and replicated timing sessions for stronger efficiency claims. This revalidation does not establish venue acceptance, originality across the literature, hiring readiness, or racial fairness.
