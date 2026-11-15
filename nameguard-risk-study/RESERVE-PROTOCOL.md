# Frozen follow-up on the ten reserve articles

The candidate-aware verifier failed its development criterion. Its registered
run remains primary for that method. Here we test a separate mechanism on the
ten unused articles, with no threshold tuning. We have seen the original
144-case development scores and the source dataset schema. No model result on
questions from these ten articles has been inspected.

Ask the frozen Qwen checkpoint for the shortest exact quotation answering the
question from the passage. Allow an explicit `UNKNOWN` answer. The proposed
candidate is never included in the extraction prompt. Greedily decode up to 32
tokens with a fresh cache. Require the decoded single line to terminate and
match the supplied candidate, after only stripping enclosing double quotes and
outer whitespace. Match is case-sensitive. Verify the proposed candidate
against its original recorded source offsets. Everything else is referred for
review. No learned adapter, answer-normalization sweep, or test-selected cutoff
is used.

Lock 160 paired paragraphs using the original source selection and ordering.
Match the four names used in the first study on 20 fixed question IDs. Compare
generated token IDs, the packed public prefix and per-step full-vocabulary
bit digests, and the deterministic evidence report directly within each
admitted name orbit. The ordinary candidate-aware, single- and two-mapping
score rules from the first study are frozen. We also run the mean-mapping rule
on the same cases and report its ranking AUC. A literal occurrence-only
baseline continues to accept every case.

Success criteria are the same 20% overall coverage, at most 10% upper exact
95% finite-population audit risk, and complete exact-name conformance. Separately
report observed errors among accepted cases and the number of supported claims
recovered. The benchmark is source support on human-annotated Wikipedia
paragraphs. The check does not establish whether real-world statements are true,
or whether a claimed CV qualification belongs to the applicant.
