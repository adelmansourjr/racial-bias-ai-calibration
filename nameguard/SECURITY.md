# Research security scope

This is an experimental information-flow prototype, not a production hiring
service or a comprehensive security product. Report suspected score-dependent
identity leakage with a minimal synthetic example and the policy/model version.
Do not attach real résumés, reviewer keys or personal applicant records to a
public issue. There is no private disclosure endpoint configured by this project.

The supported claim and explicit non-goals are in `docs/THREAT-MODEL.md`.
Reviewer keys must be randomly generated with at least 32 bytes and kept outside
the repository. Test keys are public fixtures and must never be used operationally.
The experiment generates a random key in memory and does not save it.

No blanket license is asserted for Qwen, MLX, tokenizer data or outside papers.
Model weights are not bundled. Human review and independent evaluation remain
necessary before any real hiring use.
