# P3E.2 initial-development-only real posterior-adequacy audit — 2026-08-15

Status: **protocol-valid archive; no rejection observed; posterior adequacy is
not certified and acquisition remains blocked**

This record audits the real-data archive uploaded after the frozen P3E.2
initial-development-only protocol was executed on the user's Windows working
tree. It records an archive-level evidence check and preserves the registered
claim boundary. It does not treat the uploaded output as a superiority,
held-out, discovery, or augmented-posterior-validation experiment.

## Artifact and source identity

| Item | Recorded value |
|---|---|
| Stage | `P3E.2` |
| Experiment | `real_initial_development_posterior_adequacy_audit` |
| Dataset | UCI CCPP only (`uci_ccpp`) |
| Official source path recorded by runner | `combined+cycle+power+plant\\CCPP\\Folds5x2_pp.xlsx` |
| Development rows recorded by runner | `5741` |
| Archive | `p3e_2_real_posterior_adequacy_audit.zip` |
| Archive SHA-256 | `c83d90fbc4072399839dc3cf0e7664ed6f55f01252c9ea59fdc43bfd7d98319d` |
| Archive timestamp (UTC) | `2026-08-15T09:06:15.663442+00:00` |
| Canonical source commit used for identity comparison | `7aa1b7a8ca15dff0d50e8b0e3553da891949ccfb` |
| Production-code hash in archive | `7fa31386c1571d5dbbad506ac64a66b933fbace01f5b8b54051ae252f3be4c23` |
| Canonical production-code hash | `7fa31386c1571d5dbbad506ac64a66b933fbace01f5b8b54051ae252f3be4c23` |
| Config SHA-256 in archive | `dd5659ac65393c2ffe280579e887d56ec12bc337d4dce83740111336ca138923` |
| Canonical config SHA-256 | `dd5659ac65393c2ffe280579e887d56ec12bc337d4dce83740111336ca138923` |
| Runner SHA-256 in archive | `cead87be1891d4473711b3ac5477af181b0c6940263c15955d19f67e9c7d24d7` |
| Canonical runner SHA-256 | `cead87be1891d4473711b3ac5477af181b0c6940263c15955d19f67e9c7d24d7` |

The archive contains exactly the three registered output members:
`summary.json`, `adequacy_eprocess.csv`, and `run_summaries.csv`. The
`summary.json` reports `source_hashes_verified=true`, `real_data_accessed=true`,
and zero per-run failures. Because the summary does not carry a Git commit
field, the commit identity above is established by matching all recorded
source hashes to the clean canonical tree; the official data bytes themselves
are not re-downloaded or replaced by this audit.

## Registered protocol checks

All checks below pass on the uploaded bytes:

| Check | Result |
|---|---|
| Expected/completed runs | `8/8` |
| Seeds | `2026080701`–`2026080708` |
| Rows per e-process | `97` (rounds `0`–`96`) |
| Total e-process rows | `776` |
| Domain size | `96` per seed |
| Structure/discrepancy ranks | `union_rank=19`, `discrepancy_rank=77` for every seed |
| Maximum orthogonality error | `9.06140689187345e-16` |
| Held-out opened / used for selection | `false / false` |
| Acquisition comparison / authorization | `false / false` |
| Archive rejection count | `0` |

The independent archive check also verified contiguous rounds, positive finite
e-values, `log_e_value=log(e_value)`, threshold flags, and agreement between
the CSV final/maximal values and the per-seed summary. No response-free domain
or response-order commitment was altered after execution.

## Real-data result

The registered false-alarm level is `alpha=0.01`, so the crossing boundary is
`E_t >= 100`. No seed crossed it. The per-seed terminal log Bayes factors and
maximum log e-values are:

| Seed | Final log BF | Maximum log e-value | Maximum e-value | Decision |
|---:|---:|---:|---:|---|
| 2026080701 | -0.210554640 | 0.007384347 | 1.007411679 | nominal-posterior-eligible |
| 2026080702 | -0.090854040 | 0.062440868 | 1.064431515 | nominal-posterior-eligible |
| 2026080703 | -0.069689355 | 0.141430715 | 1.151920692 | nominal-posterior-eligible |
| 2026080704 | -0.099565349 | 0.170492114 | 1.185888300 | nominal-posterior-eligible |
| 2026080705 | -0.236426241 | 0.033395411 | 1.033959298 | nominal-posterior-eligible |
| 2026080706 | 0.593213711 | 0.623535379 | 1.865511687 | nominal-posterior-eligible |
| 2026080707 | 0.104829853 | 0.115599556 | 1.122546265 | nominal-posterior-eligible |
| 2026080708 | 0.371766010 | 0.595462157 | 1.813869044 | nominal-posterior-eligible |

The largest observed e-value is approximately `1.865`, far below the
registered crossing boundary `100`. These are eight registered diagnostics,
not a newly pooled e-process; no pooled p-value, pooled Bayes factor, or
cross-seed adequacy conclusion is introduced here.

## Claim boundary and decision

The archive supports the narrow statement:

> On the registered UCI CCPP initial-development domains, across the eight
> frozen seeds, the nominal posterior was not rejected by the registered
> response-free union-orthogonal discrepancy e-process at `alpha=0.01`.

This is **non-rejection**, not a posterior-adequacy certificate. It does not
establish that the discrepancy basis is exhaustive, that its prior is
calibrated, or that the nominal posterior is safe for adaptive acquisition.
The runner's own fields remain authoritative:
`formal_real_posterior_adequacy_evidence=false` and
`formal_efficacy_evidence=false`. Gas Turbine remains outside this audit
because its existing `eta<1` branches require a separately proved
update-coherent adequacy contract.

Accordingly, the result is protocol-valid initial-development evidence that
does not trigger reference-only fallback, while the augmented posterior,
another acquisition comparison, held-out opening, superiority, motif
transfer, and scientific-discovery claims remain blocked. A separate
predictive-calibration Gate is required before any decision about downstream
acquisition is revisited.
