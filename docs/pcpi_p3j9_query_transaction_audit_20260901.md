# P3J.9 indivisible query-transaction audit (2026-09-01)

P3J.9 turns one P3J query into an indivisible response-free transaction. It
rejects a prior terminal failure, resumes an already published complete
decision without rescoring, otherwise resumes identity-bound family
checkpoints, and publishes exactly one decision. A crash after checkpoint
completion but before publication recomputes no completed grid; a crash after
decision publication reloads the decision and score audit exactly.

The persisted decision includes the complete acquisition-score record,
selected candidate and action, prior-state hash, and formal query identity. It
explicitly records that no response has been opened. Crossed or tampered
decision fields fail closed.

Any first scoring or publication error creates the no-overwrite terminal
failure and propagates. Later calls cannot retry that query. The progress
snapshot reads only model directories and checkpoint completion bits.

This Gate still does not admit a response or authorize the dataset runner.
P3J.10 must compose decision, exact matching oracle reveal, exactly-once state
advance, query-row ledger, whole-run progress, and terminal manifest checks.
