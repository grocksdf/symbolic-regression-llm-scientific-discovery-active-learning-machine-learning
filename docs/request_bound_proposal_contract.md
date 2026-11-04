# Request-bound semantic proposal envelopes

Status: code correctness, 2026-11-05; no efficacy claim.

`ProposalRuntime.propose(envelope_policy="request-bound-semantic-v1",
allow_protocol_repair=False, ...)` asks for a JSON object containing semantic
`candidates`. Required candidate fields, parent lineage, grammar, limits and
new-skeleton quota remain enforced. The caller attaches protocol/runtime/round/
island from the synchronous request rather than requiring the model to copy them.
If the model echoes any of these fields, exact value and type must agree. Candidate
parent hashes must be supplied and match; they are not silently projected. Unknown
root fields, duplicate JSON keys and contradictory metadata fail closed.

The envelope is a transport boundary, not epistemic authority. It does not change
an equation, invent a proposal, repair malformed JSON, grant admission or retry
a failed response. Localized automatic-projection contracts cannot be combined
with it. Missing administrative fields are distinct from contradictory fields.
The response hash binds the original provider content before local binding;
telemetry links that hash, prompt hash and trusted request context. This does not
cryptographically attest provider freshness when response metadata is absent.

The default `legacy` path preserves prior prompts and behavior. Synthetic tests
exercise both paths, contradictory metadata/parent rejection, malformed JSON,
single-call accounting and the prohibition on implicit repair calls.
