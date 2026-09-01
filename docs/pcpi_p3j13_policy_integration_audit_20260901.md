# P3J.13 policy integration audit

P3J.13 separates policy execution from post-transaction evaluation. Only the
registered PCPI policy may carry the four-model P3J state and enter the P3J.12
transactional measured-run path. Random, uncertainty, and QBC remain on the
unchanged legacy runner and reject any injected P3J state.

Learning curves and query audit rows are constructed only from query results
whose matching reveal has already advanced the operational state. Validation
targets enter this reporting adapter after the complete measured transaction;
they are absent from dispatch and cannot select a candidate. The failure
snapshot records identity and publication bits without response values, is
published without overwrite, and forbids seed replacement.

This stage reads no registered dataset, invokes no oracle, and provides no
formal execution configuration. Shared outer-runner wiring and the supervised
execution freeze remain blocked for P3J.14.
