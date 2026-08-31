# P3J.6 checkpointed estimator audit (2026-08-31)

P3J.6 composes the P3J.5 durable chunk protocol into one deterministic
quadrature estimator. It remains a source-correctness Gate and authorizes no
real execution.

The resumed traversal starts at the checkpoint's exact completed-action
boundary. Completed chunks are loaded and verified rather than recomputed.
Every fine and coarse grid must be complete before its score vector is
released. A refined look may reuse a preceding complete fine grid only when
quadrature order, residual-state hash, partition hash, action count, and error
safety factor all match the frozen refinement identity.

This closes the durable single-model/single-look computation boundary. It does
not yet compose all four likelihood-power models, adaptive precision looks,
representative-safe lower-envelope certification, candidate selection, or the
reveal/update lifecycle into the formal real-data runner. Those operations
remain blocked for P3J.7.
