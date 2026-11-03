# How to obtain role annotations: checks, review and honest limits

The code cannot guarantee that every person's name or indirect reference in
arbitrary raw CVs is correctly understood. It can enforce exact provenance,
review agreement and coverage of declared identity references. The distinction
is essential: signatures authenticate a judgment, not its semantic truth.

## Operational review procedure

1. Preserve the original UTF-8 source. Do not normalize, rewrite qualifications,
   remove punctuation or change line breaks when creating source offsets.
2. Ask the applicant/reviewer for the legal/display name, declared aliases,
   initials, transliterations, signatures and name-derived contact identifiers.
   Put those actual strings in a bounded registry. Do not infer a person's race.
3. `propose_candidates(source, registry)` returns lexical candidate spans marked
   `needs_semantic_review`. It does not decide identity versus evidence.
4. Inspect EVERY applicant reference and coreference in context, including those
   not matched by the registry. Add newly discovered aliases to the registry.
   Names in education, references or client descriptions may denote other people.
   Explicitly retain genuine job facts, skills, organizations and third parties.
5. Create exact identity `(start,end,text)` spans. `proposal_from_spans` reconstructs
   all original characters as typed nodes; stale, overlapping or unsorted spans
   are rejected. Mark unresolved by default. Do not infer job-criterion labels.
6. Same-spelled retained matches need exact source-bound roles and reasons. The
   retained annotations cannot overlap removed identity. This prevents silently
   erasing the programming language merely because the applicant is named Python.
7. A second actual reviewer should inspect the original source and proposed roles
   independently. Compare full payloads and role disagreement locations; adjudicate
   disagreements before setting `resolved=True`. Character agreement/kappa can
   describe consistency but is not a correctness certificate.
8. Two distinct configured credentials must endorse the exact same final payload.
   The serving authority checks both signatures, source binding, policy version,
   registry coverage and retained reasons before model execution. Changed or
   conflicting payloads, unresolved proposals and registered omissions reject.

The program enforces distinct credentials/keys, not independence of human review.
One operator can control both keys. Our tests use two machine-generated fixture
credentials and an exact generator oracle. No actual reviewer reliability study
has been conducted, and no artificial kappa=1 result is presented as human data.

## Mechanical guarantees

Every supported normalized registry match must be contained in an identity span
or a source-bound retained-role annotation. Detection checks casefolding,
full-width/compatibility forms, accents, combining marks, zero-width formatting
characters and punctuation-separated spellings. Detection never normalizes the
source fed to the model. Word-boundary checks avoid treating John inside Johnson
as a declared John reference. This is conservative matching with possible false
positives; it is not complete multilingual named-entity recognition.

`propose_candidates` can therefore return a same-spelled skill/client requiring
retention. Unknown nicknames, omitted registry entries, unsupported homoglyphs,
encoded identifiers and semantic coreferences can still escape lexical checks.
An erroneous retained-role reason signed by both reviewers can permit leakage.
The honest negative control preserves an unregistered alias Leaf; it is not
relabeled as solved.

## What correctness evidence exists here?

In 36 synthetic source documents, reviewed roles reconstruct the exact source
and agree with the generator's role oracle. Dropping each of three declared
identity nodes gives 108 controls; every control is rejected before inference.
All 36 conflicting-payload pairs are rejected. Additional CPU tests check stale
offsets, unresolved proposals, skill collisions, aliases, normalization variants,
duplicated credentials, tampering and wrong keys. These measurements verify the
implementation on known-role fixtures. They are not an estimate of actual
reviewer accuracy on real CVs.

## Independent human study needed before a stronger empirical claim

Use separately consented/deidentified CVs, an independently adjudicated oracle,
at least two reviewers, prespecified document-level sampling and a locked
annotation protocol. Report exact/overlap identity-span precision and recall,
document-level omitted-identity rate, false removal of qualification text,
adjudication rate, reviewer time and actual inter-reviewer agreement. Report
ambiguous/unresolvable cases as abstentions, not successful automatic labels.

Do not square two reported error rates without an independence justification.
Reviewers can share training, tools and blind spots. A high character-level
agreement can hide a short omitted alias in a long CV. Review quality remains
the substantive semantic bottleneck, even when token processing is efficient.

Methodological reference: [Artstein and Poesio, 2008](https://aclanthology.org/J08-4004/).

## Offline tooling

Generate an explicitly unresolved review queue:

```sh
python scripts/review_tool.py candidates --source /path/to/original-CV.txt \
  --identity 'Applicant Name' 'Declared Alias' --output review-queue.json
```

After each actual reviewer creates source-exact proposals using
`proposal_from_spans`, compare their role masks and adjudicate disagreements:

```sh
python scripts/review_tool.py compare --first reviewer-A-proposal.json \
  --second reviewer-B-proposal.json --output agreement.json
```

Each credential holder endorses the same final resolved payload with a private
raw-byte key file (at least 32 random bytes, stored outside Git):

```sh
python scripts/review_tool.py endorse --proposal adjudicated-proposal.json \
  --reviewer reviewer-A --key-file /private/path/reviewer-A.key \
  --output reviewer-A-endorsement.json
```

Repeat separately for reviewer B, then assemble the packet. Assembly itself
does not authenticate it; the serving authority verifies the two configured
signatures and all semantic-contract checks before any model computation.

```sh
python scripts/review_tool.py assemble --proposal adjudicated-proposal.json \
  --approvals reviewer-A-endorsement.json reviewer-B-endorsement.json \
  --output reviewed-request.json
```

The tool never auto-marks a review resolved or fabricates missing approvals.
The example GPU runners use synthetic credentials only and are not an actual
review workflow. Do not publish queue files containing personal applicant text.
