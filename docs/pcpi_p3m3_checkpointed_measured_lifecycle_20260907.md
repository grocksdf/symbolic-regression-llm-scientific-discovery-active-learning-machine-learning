# P3M.3 checkpointed measured lifecycle (2026-09-07)

P3M.3 closes the interruption and identity boundary left by P3M.2 without
opening a real dataset.  It introduces a P3M-only checkpoint schema rather
than changing the immutable P3J/P3K/P3L formats.

Every quadrature grid binds the exact residual-state hash, frozen partition,
predictive components, candidate-action matrix, CVaR tail probability,
quadrature order and action-chunk size.  Chunks form a SHA-256 chain and may be
appended only in one contiguous order.  Staged content is flushed and fsynced
before atomic replacement.  A proper prefix cannot release a ranking; resume
starts at exactly the first missing action.  A changed candidate coordinate,
crossed state, changed numerical plan or modified chunk fails closed.

The adaptive four-model scorer now uses these checkpoints for every P3M look.
The complete query transaction publishes a response-free decision before the
oracle is called.  Exactly one matching measured response is recorded, all
four action-conditional states advance once, and a contiguous P3M run manifest
records that held-out was not opened.  Recovery from an existing reveal
receipt reconstructs the same advance without calling the oracle again.

This is still a deterministic correctness fixture and source-composition Gate,
not an efficacy experiment.  The shared outer policy dispatcher, reporting
audit, failure snapshot, formal configuration, runtime identity and supervised
launcher are not yet frozen for P3M.  Therefore real-data and operational
execution remain unauthorized after P3M.3.
