# P3E.3 predictive-calibration compatibility audit — 2026-08-15

Status: **correctness fixture passed; real-data audit is frozen and awaits local execution**

## Purpose and boundary

P3E.3 tests whether the finite-bank posterior's sequential predictive CDF is
rejected by one fixed PIT betting family after likelihood-power selection. It is
not an acquisition experiment and is not a posterior-adequacy certificate. The
P3E.2 CCPP non-rejection result cannot be promoted to acquisition evidence
without this separate predictive diagnostic.

The protocol uses the initial-development role to select one likelihood power
and the registered validation role only for the predictive-calibration
diagnostic. It never opens the untouched-heldout role, never scores an
acquisition policy, and never changes the posterior target after seeing the
validation responses.

The PIT e-process uses three fixed mean-zero bounded basis functions (shifted
linear, quadratic, and cubic Legendre terms), four fixed betting magnitudes
`{-0.8,-0.4,0.4,0.8}`, and an equally weighted mixture. With a proper ordinary
Bayes update (`eta=1`) and conditionally calibrated continuous predictive CDF,
each betting capital has conditional mean one; the mixture is therefore a
unit-initialized e-process. Any crossing at `alpha=0.01` is evidence against
the registered calibration null for that seed, not proof that an alternative
posterior is valid.

If calibration selects `eta<1`, the run is recorded but is **not eligible for
the proper nominal-marginal interpretation** required by the P3E.2 e-process.
This is a fail-closed compatibility decision, not a license to reinterpret the
generalized posterior as ordinary Bayes.

## Frozen identity

| Item | Frozen value |
|---|---|
| Stage | `P3E.3` |
| Experiment | `real_initial_development_predictive_calibration_audit` |
| Dataset | UCI CCPP only (`uci_ccpp`) |
| Replicates | `2026080701`–`2026080708` |
| Initial development budget | 32 rows per seed |
| Validation diagnostic budget | 256 rows per seed |
| Power candidates | `(0.125, 0.25, 0.5, 1.0)` |
| Power selection | Prequential posterior-randomized R-log SafeBayes; development responses only |
| Transform | Standardizer fit on the 32 initial-development rows only |
| Bank/preconditioner | `generic-real-bank-v1`; termwise center/scale from initial covariates |
| PIT method | `prequential-pit-mixture-e-process-v1` |
| False-alarm level | `alpha=0.01`, threshold `E_t >= 100` |
| Validation order | Stable SHA-256 row-ID order, fixed before validation responses are used |
| Held-out | Closed; never opened or used for selection |
| Acquisition comparison | Not run |
| Acquisition authorization | Blocked |

## Required output

The local runner must produce:

- `summary.json`, with source hashes, split commitments, selected power,
  validation/PIT decisions, and the explicit claim boundary;
- `calibration_scores.csv`, one row per seed and candidate power;
- `pit_eprocess.csv`, rounds `0` through `256` for every seed;
- `run_summaries.csv`, one row per seed.

The summary must retain `heldout_opened=false`,
`selection_used_heldout=false`, `acquisition_comparison_performed=false`,
`acquisition_authorized=false`, and
`formal_predictive_calibration_evidence=false`. A complete run with no PIT
crossing remains non-rejection only; no pooled cross-seed e-process is added.

## Local Windows command

```powershell
$project = "D:\01\666\hypothesis_mvp_p3e2_canonical"
$dataRoot = "D:\01\666\hypothesis_mvp\data"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
$output = "$project\outputs\p3e_3_real_predictive_calibration_audit"

Set-Location $project
& $python -B scripts\run_pcpi_p3e3_real_predictive_calibration_audit.py `
    --data-root $dataRoot `
    --output-dir $output `
    --config configs\p3e_3_real_predictive_calibration_audit.json `
    --phase P3E.3 `
    --heldout-state closed
```

Do not change seeds, budgets, candidate powers, response order, roles, or
configuration fields. If the command exits nonzero, upload the complete output
directory and traceback; do not edit the summary manually.
