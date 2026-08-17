# P3F.2a--c open-target correctness result — 2026-08-16

Decision: **PASS FOR MATHEMATICAL/IMPLEMENTATION CORRECTNESS ONLY.**

The runner `scripts/run_pcpi_p3f2_open_target_correctness.py` used only a
hand-constructed algebraic exact-reference fixture. It did not read a real or
generated dataset, did not open held-out data, did not execute an acquisition
policy, and did not produce calibration, efficacy, discovery, or law evidence.

## Frozen reference

- stage: `P3F.2`, subgates `P3F.2a`, `P3F.2b`, `P3F.2c`;
- full prior: countably-open typed finite-AST target;
- exact slice: node count at most three, conditional prior mass `0.936`;
- explicitly omitted full-target tail mass: `0.064`;
- raw AST states: `14`;
- exact polynomial equivalence classes: `8`;
- collapsed generative states: `42`;
- proposals: `complete-uniform` and `prior-independence`;
- real-data access: forbidden;
- held-out role: not applicable.

## Gate result

All 15 registered decisions passed:

- open-prior normalization and explicit nonzero tail;
- response-independent target contract;
- joint probability normalization;
- exact equivalence-class mass conservation;
- raw-AST/scientific-class separation;
- batch/sequential component identity;
- exhaustive sequential-SMC/batch identity;
- prequential evidence telescoping;
- RJMCMC detailed balance and stationarity;
- proposal invariance; and
- row-order equivariance.

Maximum observed numerical errors were:

| Diagnostic | Maximum error |
|---|---:|
| prior normalization | `0` |
| joint probability normalization | `1.1102230246251565e-16` |
| equivalence mass conservation | `0` |
| batch/sequential identity | `5.329070518200751e-15` |
| evidence telescoping | `5.329070518200751e-15` |
| RJMCMC detailed balance | `6.938893903907228e-18` |
| RJMCMC stationarity | `1.1102230246251565e-16` |
| cross-proposal posterior difference | `0` |
| row-order equivariance | `4.440892098500626e-16` |

Frozen config SHA-256 at the passing run was
`23137bde21ab790495336766cec870e4e17c3319a89f1076372445d834a31727`.
The target-contract hash was
`7207d25689341870501fc0ac45627ff9729f34f5efe1f5a2ffcfb434328835f9`
and the grammar hash was
`588682a9ede1a47ab4a90f8197578ea2e9de87bf38095bf78ab1ac8ae7aa3330`.
Production-code and runner identities will be regenerated after the final
documentation and manifest synchronization; the final delivery manifest is
authoritative for those whole-tree identities.

## Claim boundary

This result proves that the registered exact reference implements its stated
probability target and reversible updates to floating-point tolerance. It does
not prove that the small grammar contains a real law, that a scalable particle
algorithm explores the open target, that the predictive law is calibrated, or
that PCPI outperforms any method. Acquisition and held-out confirmation remain
blocked.
