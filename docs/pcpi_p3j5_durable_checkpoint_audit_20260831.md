# PCPI P3J.5 durable checkpoint audit

P3J.5 supplies a response-free checkpoint protocol for the contiguous action
chunks introduced by P3J.4. It does not compose the measured-data runner and
does not authorize operational execution.

Every plan binds the complete predictive-component arrays, frozen partition,
class-conditional residual state, node order, action count and chunk size. A
checkpoint is a hash-chained contiguous prefix beginning at action zero. Every
append must be exactly the next registered chunk and must carry the same state,
partition and precision identities. Gaps, duplicates, crossed plans and stale
decisions fail closed.

Publication creates a new same-directory staging file exclusively, writes one
canonical JSON payload, flushes and `fsync`s it, then atomically replaces the
checkpoint. A surviving staging file is not overwritten or silently accepted.
The final checkpoint may be loaded only after its complete chain, per-chunk
hashes, plan identity, score finiteness and terminal completeness flag verify.

No score vector is released to ranking or candidate selection from a proper
prefix. Correctness tests verify two-chunk interruption/recovery against the
direct batched result, and reject out-of-order, duplicate, altered-state,
altered-predictive-component and tampered-score inputs.

P3J.5 passes as `passed-durable-checkpoint-runner-composition-blocked`.
The next Gate may compose this checkpoint protocol with the supervised runner,
but must keep the existing pre-data authorization guard, write progress outside
formal result tables, and require all model/precision/action chunks before one
candidate can be selected.
