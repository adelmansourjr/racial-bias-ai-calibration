# Independent annotation and task-validation protocol

Status: prepared, not conducted. No independent-human result is claimed.

Recruit at least two reviewers who did not implement the method or write its
fixtures. Record relevant expertise, conflicts, compensation and review dates.
Do not give reviewers model scores, which method admitted an input, or the
generator's proposed spans. Obtain permission for any nonpublic applicant data;
the included packet contains only original synthetic examples.

For each case, independently mark applicant-identity character intervals, mark
name-like text that must remain as job evidence, and select whether the explicit
qualification condition is met. Mark ambiguous cases as uncertain and explain
what additional context is needed. Character offsets are Python Unicode
code-point offsets, half-open [start,end), over the exact text in cases.json.
Never silently normalize or edit that source.

After independent submissions are frozen, compute exact span agreement,
character-level overlap, uncertainty rates and qualification agreement. Report
disagreements before adjudication. An adjudicator then resolves disagreements
using declared role definitions; retain both original submissions. Two credential
signatures are authentication mechanics, not evidence that this review happened.

No fixed sample-size claim is made for this 24-case pilot. A subsequent real-CV
study should recruit job-domain reviewers and predeclare task labels, recruitment,
protected-attribute data governance, sample size and a power analysis at the CV/job
unit. Estimate role reliability and admission differences by actual reviewed
naming conventions. Do not infer race from script or surname alone.

Use two copies of blank-response.csv, with different reviewer IDs. Fill
identity_spans_json with a JSON array of [start,end] pairs, retained_spans_json
similarly, status with clear or uncertain, and eligible with yes/no/uncertain.
The analyzer requires explicit independent-review attestation in both files,
checks exact case coverage and source binding, and refuses blank templates.
It cannot independently verify the truth of that attestation.

Example command after actual review:

    python3 review-packet/analyze_reviews.py reviewer-a.csv reviewer-b.csv --output review-results.json

The project author should prepare an ownership statement explaining which
mathematical arguments, experiments and implementation decisions they can
independently reproduce and defend. The current work used substantial AI
assistance for code, analysis and writing; edit that disclosure to the actual
contribution history and the target venue's policy before submission.

## Clarification after the first AI round

The original reviews independently identified an unspecified convention for
uncertain spans. In future rounds, annotate only roles established by the
available text. Leave an unresolved role unassigned, mark status uncertain and
explain the missing context in notes. Do not assign a tentative role as though
it were approved. Retain any other role that is actually established. This
clarification was added after the first AI annotations were sealed; their
agreement is not evidence that this later convention was preregistered.
