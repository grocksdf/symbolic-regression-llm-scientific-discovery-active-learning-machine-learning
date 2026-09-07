# P3M.4 outer policy and reporting correctness (2026-09-07)

P3M.4 routes only the registered PCPI policy through the P3M transaction;
random, uncertainty and QBC remain on their existing matched-budget paths.
Any P3M failure publishes its own immutable snapshot and forbids retry or seed
replacement.

Validation metrics are computed only after the complete reveal ledger.  The
report audit requires the P3M action-conditional joint identity, the exact P3M
CVaR method, the complete four-power lower envelope, valid representative-set
selection, certified/frozen interval resolution, and no conditional EPIG or
information-invariance shortcut.  Mean information and negative-gain mass are
retained as diagnostics; CVaR is the selection score.

The outer composition now identifies P3M separately through acquisition,
post-ledger evaluation and summary.  Correctness fixtures verify a decision-
rule-valid rate of one without accessing a real dataset.  Formal configuration,
runtime binding, execution supervisor and one-time output identity remain to
be frozen.  P3M.4 does not authorize real or operational execution.
