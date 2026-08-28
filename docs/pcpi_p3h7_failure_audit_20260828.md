# PCPI P3H.7 incomplete-run failure audit

P3H.7 was stopped by the user after its registered validity condition had
already become impossible. It is an incomplete, protocol-invalid development
run and is not efficacy evidence. It must not be resumed, completed, pooled
with a later protocol or described as a successful formal experiment.

The preserved append-only progress log has SHA-256
`314c4763c099e961abf81d2e0ff987698cf2611ffd580811c4f9f3527f5f5cc1` and
contains 3,027 events: 92 policy starts, 87 policy completions, four terminal
policy failures and 2,840 completed acquisition queries. Execution stopped in
run 92/96, NOx seed `2026080707`, after query 14/32. No terminal summary,
manifest or efficacy assessment was published.

All four terminal failures were PCPI runs:

- CCPP, seed `2026080708`;
- Gas Turbine CO, seeds `2026080701` and `2026080708`;
- Gas Turbine NOx, seed `2026080702`.

Each failed because the transformed maximin-utility intervals still overlapped
at the frozen maximum quadrature order. The implementation correctly refused
to reveal a response and did not retry, replace a seed or fall back to
Student-t EIG, posterior variance or QBC. The failure is therefore a defect in
the totality of the operational numerical decision rule, not a leakage defect.

P3H.8 repairs the decision rule generically. The failed dataset and seed names
are not inputs to that rule, no observed gain or validation metric sets a
threshold, and the P3H.7 terminal-abstention behavior remains available under
its historical protocol identity.
