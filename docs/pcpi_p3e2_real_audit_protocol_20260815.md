# P3E.2 initial-development-only real posterior-adequacy audit protocol

Status: **frozen protocol; archive-level real result audited 2026-08-15**

The uploaded archive is audited in
`docs/pcpi_p3e2_real_posterior_adequacy_result_20260815.md`. The audit found
8/8 completed CCPP runs, zero threshold crossings, and no held-out or
acquisition access. This status records protocol-valid non-rejection only; it
does not certify posterior adequacy or authorize a downstream experiment.

## Scope

This stage audits the declared finite-bank posterior against one registered
orthogonal discrepancy alternative on official measured data. It is a
posterior-adequacy diagnostic, not an acquisition experiment. The runner does
not compare policies, does not reveal acquisition-pool responses, and does not
open the untouched-heldout role.

The audit is deliberately asymmetric in its interpretation:

- non-rejection of the nominal model is not an adequacy certificate;
- rejection against the registered discrepancy alternative is evidence against
  nominal adequacy for that target/seed, but does not validate the augmented
  posterior for acquisition;
- any downstream acquisition run remains blocked until this diagnostic and a
  separate predictive-calibration Gate are reviewed.

Gas Turbine CO/NOX are intentionally outside this first audit. The existing
real protocol selected `eta<1` for most Gas target/seed calibrations, while the
P3E.2 Ville/e-process guarantee is stated for a proper nominal marginal.
Including those targets here would silently change the null and invalidate the
claim boundary; a Gas audit requires a separately proved update-coherent
adequacy contract.

## Frozen identity

| Item | Frozen value |
|---|---|
| Stage | `P3E.2` |
| Experiment | `real_initial_development_posterior_adequacy_audit` |
| Config | `configs/p3e_2_real_posterior_adequacy_audit.json` |
| Official target | UCI CCPP (the confirmed `eta=1` blocker) |
| Split | Existing P2A role protocol, split seed `20260807` |
| Replicates | `2026080701`–`2026080708` for CCPP |
| Domain size | 96 initial-development rows per target/seed |
| Domain selection | SHA-256 row-ID order; covariate/identifier only |
| Response order | Separate SHA-256 row-ID order; predictable before responses |
| Target transform | Initial-development-only standardization |
| Structure bank | `generic-real-bank-v1`, six finite structures |
| Design transform | Termwise center/scale from development covariates only |
| Discrepancy basis | Response-free union-orthogonal RBF basis |
| E-process | Prequential orthogonal-discrepancy Bayes-factor e-process |
| False-alarm level | `alpha=0.01`, threshold `E_t >= 100` |
| Held-out | Closed; never opened or used for selection |
| Acquisition comparison | Not run |
| Acquisition authorization | Blocked |

## Statistical object

For each CCPP seed, the 96-row domain is fixed from initial-development
row identifiers before any response is passed to the adequacy engine. Let
`D_s` be the six registered structure designs on this domain and let `B` be the
response-free RBF basis projected onto the complement of the union of their
spans. The audit compares

\[
M_0: y=D_s\beta_s+\varepsilon,
\qquad
M_1: y=D_s\beta_s+B\gamma+\varepsilon,
\]

with proper conjugate Gaussian–inverse-Gamma priors and the same structure
prior under both alternatives. The full e-process is recorded from round 0
through round 96. A crossing triggers the registered reference-only decision
for that target/seed; it does not authorize an augmented-posterior run.

## Required output

The local run must produce:

- `summary.json`, including source hashes, split commitments, every target/seed
  decision, and the explicit claim boundary;
- `adequacy_eprocess.csv`, containing every e-process value and threshold check;
- `run_summaries.csv`, one row per target/seed with final log Bayes factor,
  maximum e-value, crossing round, and decision mode.

The summary must state `heldout_opened=false`,
`selection_used_heldout=false`, `acquisition_comparison_performed=false`, and
`acquisition_authorized=false`. A successful protocol run still does not make
`formal_efficacy_evidence` true.

## Local Windows command

```powershell
$project = "D:\01\666\hypothesis_mvp_p3e2_canonical"
$dataRoot = "D:\01\666\hypothesis_mvp\data"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
$output = "$project\outputs\p3e_2_real_posterior_adequacy_audit"

Set-Location $project
& $python -B scripts\run_pcpi_p3e2_real_posterior_adequacy_audit.py `
    --data-root $dataRoot `
    --output-dir $output `
    --config configs\p3e_2_real_posterior_adequacy_audit.json `
    --phase P3E.2 `
    --heldout-state closed
```

Do not rerun with changed seeds, budgets, data roles, response order, or
config values. If the command exits nonzero, upload the complete output
directory and terminal traceback; do not manually edit the summary.
