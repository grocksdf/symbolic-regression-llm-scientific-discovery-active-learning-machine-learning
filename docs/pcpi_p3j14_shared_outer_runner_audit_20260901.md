# P3J.14 shared outer-runner audit

P3J.14 composes the complete P3J policy path without copying the established
baseline runner. The registered PCPI policy enters P3J.12 transactional
acquisition, then P3J.13 post-ledger reporting and matched summary construction.
Random, uncertainty, and QBC call the existing `run_pcpi_p3b_real._run_policy`
implementation unchanged. Any exception across transactional acquisition,
reporting, or summary publishes the no-response-value P3J policy failure
snapshot before propagating.

The full configuration freezes the original datasets, eight seeds, 32+32
measurement budgets, 128-candidate domain, 256 validation observations, four
likelihood powers, quadrature schedule, action chunking, and all P3J protocol
identities by a byte-exact SHA-256 digest.

Both operational and formal dataset authorization remain false. The runner
validates the frozen configuration and raises before checking a data root,
creating an output directory, or calling a dataset loader. No efficacy result
is produced. Supervised restart/process monitoring and final execution
authorization remain for P3J.15.
