# PCPI P3J.4 batched class-density audit

P3J.4 replaces the per-action SciPy distribution dispatch path with bounded,
contiguous action batches. It is a response-free computational composition:
no probability model, quadrature node, 64-step inverse-CDF bracket, `2e-13`
inversion tolerance, class interaction, error envelope or selection rule is
changed.

For each source class, raw-PIT quadrature nodes are shared across an action
chunk. Component-mixture inversion is evaluated as one tensor with axes
`component x action x node`. The resulting responses are then evaluated under
every target-class density in the same action order. Chunk outputs are written
back to their original contiguous positions; candidates are never reordered.

The correctness fixture compares the batched kernel at chunk sizes 1, 2 and 7
against the historical scalar-action kernel. Every class marginal is identical
and mutual information agrees within `2e-15`. Invalid chunk sizes fail before
evaluation.

With 128 candidates, chunk size 16, four ambiguity models, seven classes and
the six actual grids remaining after P3J.3 reuse, the static dispatch ledger is:

- scalar SciPy distribution calls: `1,720,320`;
- batched calls: `107,520`;
- saved calls: `1,612,800` (`93.75%`).

This is a dispatch/vectorization improvement, not a claim that the underlying
`101,154,816` scalar cross-class density evaluations disappeared. The chunk
size is an implementation memory bound and is not selected from efficacy or
timing outcomes.

P3J.4 passes as `passed-batched-density-kernel-runner-checkpoint-blocked`.
Runner composition and real execution remain false until deterministic chunk
progress can be fsync-published, validated and resumed without recomputing or
selecting favorable partial results.
