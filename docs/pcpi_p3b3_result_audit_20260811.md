# P3B.3 returned-result audit — 2026-08-11

## Identity and protocol

The audited result is
`p3b_3_real_matched_budget_acquisition_uci_mainline_20260811_230224.zip`.
Its run manifest resolves to source-tree hash
`33a903398c9383254ea8eb4cd6f8e60d250032a8c06b8b73cb89e9e41ec20577`
and production-code hash
`2a3442d4bccc1ebaec594e302e944618510f0bf10da39c45fc8ced8e31102e95`.
The exact formal source package hash is
`bf41cb7d3d8a742970dce9f01945e324b54b998fe45219d99c3478d2e8cf4d6e`.

The EvidenceRegistry chain is valid: 97 events, head hash
`3c5ad091214a1193949dc0f5885f3c839d9fea8d2d40004b08a45d881ae4c5b4`.
All nine registry exports match their committed hashes. The run contains 96/96
policy runs, 3,168 learning-curve rows, 3,072 query rows, and zero failed runs.

All registered protocol checks passed. Initial, acquisition, validation, pool,
candidate-evaluation, policy, seed, split, and source budgets match. CCPP and
Gas CO/NOX use shared family accounting. Untouched-heldout remained closed and
selection did not use heldout.

## Independent family-level reproduction

Negative nAULC delta favors PCPI; positive frozen-class-gain delta favors PCPI.

| Family | Baseline | Mean nAULC delta (95% CI) | Mean class-gain delta (95% CI) |
|---|---:|---:|---:|
| CCPP | random | -0.0230 (-0.0550, 0.00891) | approximately 0 (-1.08e-15, 8.59e-16) |
| CCPP | uncertainty | 0.000418 (crosses 0) | approximately 0 (crosses 0) |
| CCPP | QBC | -0.00477 (crosses 0) | approximately 0 (crosses 0) |
| Gas family | random | -0.00756 (-0.0697, 0.0545) | 0.0209 (-0.0275, 0.0694) |
| Gas family | uncertainty | 0.00302 (crosses 0) | 0.0142 (crosses 0) |
| Gas family | QBC | -0.0164 (crosses 0) | -0.00144 (crosses 0) |

CCPP final RMSE and NLL improve significantly against random, but match the
uncertainty policy essentially exactly. Gas final RMSE and NLL means are worse
than random, with intervals crossing zero. These endpoint observations do not
override the preregistered family-level acquisition Gate.

## Posterior and acquisition diagnostics

CCPP initial frozen-class entropy has median `4.4546e-13` and range
`[9.32e-15, 3.68e-11]`. It always has two operational classes, but certified
class EIG is used for only 1.17% of PCPI queries; 98.83% use the explicit
epistemic fallback.

For Gas CO the median initial entropy is `5.28e-4`; for Gas NOX it is
`4.82e-3`. The grouped Gas class-EIG use rate is 56.05%. The PCPI
decision-rule-valid rate is 1.0 in both families.

PCPI averages 51.0 seconds per run versus about 0.27 seconds for each baseline,
approximately 187 times more wall time. No failed seed is hidden.

## Gate decision

- Protocol Gate: **PASS**.
- Decision-rule correctness in the returned real run: **PASS**.
- Real family-level efficacy Gate: **FAIL**.
- Submission Gate: **NOT YET**.
- P4/P5 progression: **BLOCKED**.

The result supports a valid negative real-development conclusion and motivates
posterior/model repair. It does not support acquisition superiority,
open-grammar discovery, held-out confirmation, physical intervention, motif
safety, VED discovery, or a new law.
