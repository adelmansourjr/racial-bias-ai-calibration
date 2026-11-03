# Threat model and trust boundary

The adversary controls request bytes and may modify original identity strings,
insert unusual names, attempt role/text tampering after review, add JSON fields,
repeat keys, submit unsupported lengths or issue overlapping requests. The
operator configures a trusted reviewer-key map, checkpoint manifest, pinned
base model and scoring policy. Review signing is an operator/reviewer workflow,
not an applicant-accessible signing endpoint.

Trusted components: role judgments and review keys; operator configuration;
the scalar tokenizer/codec; Python and HMAC implementation; frozen Qwen modules;
MLX and deterministic kernels; trusted host memory; and name-independent external
slot assignment. HMAC proves that a configured key holder issued the same bytes;
it neither proves role completeness nor authenticates a real human by itself.

The serving API returns only `PublicScore`. A separate trusted research backend
subclass observes tensors for experiments. There is no user-controlled audit
flag. Python privacy conventions are not an isolation boundary against code
already executing in the host process; such code could inspect `_model`, keys
or buffers and lies outside the claim.

Model parameter content is checked on load and final verification. Each inference
checks graph and parameter-object identity before/after, rejects known overlays,
uses a fresh cache-free computation and acquires a process-wide model lock.
MLX array immutability and a benign host are assumptions. These checks are not
hardware attestation or isolation from malicious processes.

Different rejection outcomes, admission limits, wire sizes, codec cache timing,
compute time and memory are observable. This work does not establish constant-time
or termination-sensitive confidentiality. Changing names can still change
those observations. The public-score theorem covers two requests that are both
accepted under the same trusted configuration.

Unsupported inputs must be rejected, not silently truncated or assigned a fake
neutral score. A model failure poisons the service; rebuild before reuse.
Nonblocking busy rejection in the backend is conservatively treated as such a
failure. This prototype favors rejection over concurrent throughput.

Identical public margins tie by stable integer slots in the provided helper.
If an external system derives those slots from names or uses identity metadata
to break ties, that system violates the integration premise. The scorer cannot
control arbitrary downstream applications.
