# Hardened native-token name-channel scoring

This revision preserves all 192 previously measured CV scores and 5,952 aligned
activation fingerprints. It adds fail-closed lifecycle handling, a fixed
credential-task readout, explicit whitespace review proposals and a contract
binding interchangeable requests to the exact same evidence-token sequence.

| Check | Result |
|---|---:|
| Previously measured CV scores preserved exactly | **192/192** |
| Previously measured activation fingerprints preserved | **5,952/5,952** |
| Zero identity-slot stage checks | **5,760 passed** |
| Clean native full-vocabulary comparisons | **32/32 exact** |
| Held-out synthetic credential decisions | **48/96 → 96/96** |
| Identical calibrated scores across names | **24/24 condition groups** |
| Correct qualification-change direction | **48/48 pairs** |
| Backend failure, reuse, close and fresh recovery | **Passed** |
| CPU contract, lifecycle and mathematical tests | **29 passed** |

The 96 credential rows contain six held-out credential types, two instruction
styles, two qualification states and four names. They represent 24 condition
groups, with repeated names as counterfactual variants. The threshold was fitted
on 32 rows from eight different credential types and one fitting name, then
frozen before these held-out calls. The credential patterns revisit the earlier
project rubric. They are held out from this threshold fit, rather than an
independent human hiring study or a newly untouched benchmark.

The fixed threshold is 4.733816146850586 A-minus-B logit units. It applies only to
the named synthetic credential profile. Original CV/job readouts stay available
with their exact previous scores. The conditional A probability remains a
normalized model readout, with no claim that it estimates real job success.
Threshold tie-breaking uses the finite candidate grid recorded in the source;
it does not claim a minimum-shift optimum over every real-valued threshold.

## Specific problems fixed

The earlier scorer could be reused after a backend error. The new service
records a poisoned state, suppresses private exception details and rejects
reuse. Malformed requests remain recoverable and are checked before model
execution. A per-service request lock complements the shared model lock, and
close is idempotent. Tests cover overlapping requests, global contention,
invalid packets, post-failure reuse and normal reconstruction.

The rubric had correct qualification directions but a poor zero-threshold
decision rule. A fixed scalar task calibration corrects this measured offset.
Because the same constant is used for all admitted names, it preserves exact
name equality. This readout adjustment is separate from the internal identity
activation gate. The calibration does not supply identity detection or erase
representations. Its artifact is bound by a SHA-256 check and an exact schema.

The boundary helper proposes only adjacent whitespace reclassification under an
explicit policy. It preserves the source text and original tokenizer IDs,
reports the expanded span and returns an unresolved proposal. A reviewer must
approve it before ordinary authenticated scoring. The normal preparation entry
point first preserves already valid original roles, including long formatting
runs, before attempting optional repairs.

See [the detailed issue ledger](ISSUES.md).
The [formal claim and study protocol](METHOD.md) give the proof, observer boundary,
experimental units, prior literature and remaining validation requirements.

## Boundary compatibility and evidence equality are distinct

In the same 120 synthetic tokenizer cases, strict compatibility accepted 87.
Permitted whitespace proposals increased token-compatible cases to 112.
However, some repairs change the retained evidence-token sequence. Treating
all 112 as interchangeable would be incorrect.

The new EvidenceContract stores the immutable approved evidence-token sequence
and compares it directly. SHA-256 is a reporting identifier, so the equality
guarantee does not require an assumption about hash collisions. Against the
first approved anchor in each layout, 103 of 120 cases
matched the evidence contract. Nine token-compatible cases were rejected
because evidence differed, and eight failed token compatibility. These
anchor-conditioned counts differ from simple tokenizer admission counts.

The contract guards claims of equivalent inputs. Different repairs can
legitimately receive different reviewed evidence contracts. An actual
qualification edit requires a new reviewed CV/job contract. Admission and
timing can still vary by name. The score guarantee concerns admitted requests
with the same evidence contract.

[Boundary audit](results/boundary-contract-exact-audit.json)

## Runtime architecture and mathematical preservation

The original named source is tokenized and embedded intact. Designated identity
rows are set to zero inside the model. Each native transformer block gathers the
ordered evidence rows, computes them at evidence-relative positions and scatters
them back into the original physical slot layout. This retains the successful
shape-stable recurrence of the previous experiment.

For evidence gather G and scatter S, G(S(x))=x. The recurrence

~~~text
H_(l+1) = S(B_l(G(H_l)))
~~~

depends only on the ordered evidence IDs and fixed model parameters. Equal
approved evidence produces equal scoring tensors under the pinned deterministic
runtime. The optional credential readout is

~~~text
m_credential = m_original - threshold
~~~

so equal original margins stay equal after the fixed task transformation.
The interface rejects an incoming variant whose evidence IDs differ from
the operator-approved contract. Identity rows remain zero at each stored block
boundary. Original raw embeddings and token geometry remain private host
observations, and the active block computation uses gathered evidence rows.

The 192-case regression compares scores and every aligned stage fingerprint
through the public score method. It reproduces the earlier 32 CV/job contexts
under all six names, including variable native name lengths. An independent
service smoke test combines the saved task calibration and evidence binding on
the actual GPU, checks short/long name equality and rejects two stale evidence
contracts.

## Use the service

From the repository root:

~~~sh
python3 -m unittest discover -s native-token-gate-hardening/tests -v
python3 native-token-gate-hardening/verify.py
~~~

For a fresh GPU check, use output paths that do not already exist:

~~~sh
python3 native-token-gate-hardening/boundary_audit.py --output /tmp/new-boundary-audit.json
python3 native-token-gate-hardening/integration_smoke.py \
  --checkpoint /path/to/pinned/qwen --output /tmp/new-service-check.json
~~~

The main experiment deliberately refuses to overwrite its completed plan and
records. Repeating that private-data regression requires a fresh local sibling
study directory and the earlier local records identified by the plan hashes.

GPU execution requires the pinned environment in
[requirements-mlx.txt](../nameguard-validation/requirements-mlx.txt) and the exact
checkpoint described in audit/checkpoint-manifest.json. Supporting model and
authentication code is reused from nameguard-validation/src/nameguard, with
dependency hashes in the plan.

A raw named prompt can be scored using reviewed Unicode-character spans:

~~~sh
python3 native-token-gate-hardening/demo.py \
  --checkpoint /path/to/pinned/qwen \
  --prompt-file native-token-gate-hardening/examples/credential-negative-short.txt \
  --span 16:18
~~~

For the fixed credential task, add --credential-profile. The runtime verifies
the published threshold artifact before using it. This profile expects the
synthetic active-credential task described by the experiment. Operator selection
of that profile is explicit. Arbitrary hiring prompts need separate rubric
validation.

The API composes a fixed profile with an evidence contract as follows:

~~~python
from evidence_contract import EvidenceContract, BoundEvidenceScorer
from service import CredentialScoringService

contract = EvidenceContract.from_reviewed(approved_wire, authority, tokenizer)
bound = BoundEvidenceScorer(raw_scorer, contract)
credential_service = CredentialScoringService(bound, fixed_calibration)
result = credential_service.score(reviewed_variant_wire)
~~~

The approved evidence-token tuple is trusted operator configuration. Incoming
requests cannot replace it. Runtime failures require a fresh scorer.
Preparation uses prepare_review.propose_whitespace_repair, whose output stays
unresolved until reviewed.

The CLI is a local demonstration: supplied spans represent the operator's
review, and temporary keys exercise the authentication mechanics. It does not
claim two independent human reviews. The included short/long synthetic prompts
have 76/82 original native tokens and 75 evidence tokens each. Separate CLI
processes returned exactly the same calibrated margin, -3.9113502502441406,
and probability. Use span 16:41 for the long example. These two fresh-process
measurements are preserved in results/demo-short.json and results/demo-long.json.

## Research record and practical limits

The plan freezes four experiment source files and dependency hashes before
inference. The calibration artifact records its selected threshold before
held-out inference. Later wrappers, the review-entrypoint correction, evidence
binding and their tests are covered separately by the final public manifest.
The second boundary audit follows the first sweep's discovery and is explicitly
a follow-up. The initial supplemental scripts and results are preserved under
audit/initial-supplemental and results. The final exact-comparison contract was
then rerun through the 120-case tokenizer sweep and the four-forward GPU service
smoke test. Earlier completed studies remain preserved.

The public release includes source, aggregate outputs and the synthetic fitting
threshold. Actual CV/job text, per-CV records, review keys, hidden states and
original model weights remain local. Public verification checks source integrity
and aggregate consistency. When local credential rows are available, it also
reconstructs the threshold arithmetic, accuracy, name equality and paired
qualification effects. It does not independently re-run GPU inference.

The role contract remains essential. It cannot infer every unknown alias or
resolve an ambiguous name reference using missing outside context. Other
demographic proxies and job-policy choices may remain in retained evidence.
The result is exact isolation of designated name fields under the admitted
evidence contract. Hiring validity, independent semantic annotation and broader
demographic outcomes still require separate empirical evidence.

Zhao et al. provide a precedent for correcting answer-label priors. Their
content-free calibration differs from the labeled threshold fit here.
[Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)

The existing span-veto and information-flow literature motivates keeping the
policy boundary and deterministic execution assumptions explicit.
[What the Model Cannot See](https://arxiv.org/abs/2610.15883) ·
[Language-Based Information-Flow Security](https://www.cs.cornell.edu/andru/papers/jsac/sm-jsac03.pdf)
