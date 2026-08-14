# Hypothesis MVP — PCPI

Current stage: **P3 repair required; P3C.1 efficacy NO-GO**. P3B.10 and P3C.1
discrepancy-aware repair both completed 96/96 protocol-valid real runs but
returned `REAL_ADVANTAGE_NOT_DEMONSTRATED`. P3C.1 preserves the frozen P3B.10
data, splits, budgets, baselines, nominal reporting posterior, predictive
classes, likelihood-power ambiguity set, representative safe set, and closed
held-out state; it adds only a response-free predictive discrepancy envelope to
PCPI candidate ranking. The P3C.1 audit is recorded in
`docs/pcpi_p3c1_result_audit_20260813.md`. No P4/P5 or superiority claim is
authorized. The task-independent P3D.1 decision repair has passed its isolated
correctness Gate but is not integrated into the real runtime; no new real-data
run is authorized by this source.

Source synchronization warning: the public clean import at base commit
`5fa4f3c9080c31208135dcada80a8f86121a199a` omitted the
`hypothesis_mvp.data` Python package even though the imported delivery manifest
lists it. Full regression and all real entrypoints are therefore blocked until
the six provenance-verified files are restored. See
`docs/pcpi_public_source_sync_audit_20260814.md`.

PCPI treats symbolic scientific discovery as sequential Bayesian discrimination
among operational predictive-equivalence classes. The target is the joint
posterior over symbolic structure, coefficients, and observation noise.

## Evidence status

The original registered-real P2A run passed its finite-bank exact-posterior
numerical checks but showed severe long-horizon root-ancestor coalescence for
Gas Turbine CO/NOX. P2A.1 replaces plug-in particle likelihood weights with an
independent Rao–Blackwellized structure update, adaptively tempers informative
observations by conditional ESS, applies target-invariant moves at every
bridge, and reconstructs coefficient/noise particles from the exact final
conditional posterior.

Historical generated-observation efficacy runs remain revoked. Controlled
fixtures may enter the separately labelled correctness evidence namespace for
P1/P2A.1/P2B/P3A.2 only. They may never enter the real-data efficacy namespace or
support a scientific-discovery advantage, new-law, or real-measurement claim.
The P2A.1/P2B/P3A.2 numerical fixture is isolated under
`pcpi/reference/inference_fixture.py` and is never imported by real-data
discovery or acquisition runtimes.

The formal P2A.1 correctness entrypoint is `pcpi-p2a1-diagnostic`. It has no
data-root option and validates the single canonical SMC engine against an exact
posterior. The heldout-closed real calibration entrypoint remains
`pcpi-p2a-real`. It accepts:

- `uci_ccpp`;
- `uci_gas_turbine_co`;
- `uci_gas_turbine_nox`.

CO and NOX are two targets from one Gas Turbine family. Airfoil remains a
frozen development pilot. NIST ASD, VED, motif transfer, and held-out
confirmation remain gated.

P2B adds one explicit reversible proposal catalog over a closed diagnostic
bank. Birth, death, and equal-dimensional replacement moves record forward and
reverse probabilities. The collapsed MH correction uses their ratio and a
unit Jacobian because coefficients and noise are analytically integrated until
the final conditional reconstruction. No LLM or motif enters this kernel.

P3B.1 introduced classes defined by a deterministic complete-link partition of
uncertainty-standardized posterior-predictive quantile distances on a frozen
operational action domain. P3A.2 validates this same repaired definition by
independent high-precision adaptive quadrature and a Student-t-measure
Gauss--Jacobi estimator over eight posterior-concentration scenarios and five
evaluation budgets. The nested fine/coarse discrepancy supplies an explicitly
asymptotic numerical-error envelope; it is not a probabilistic confidence
interval. P3B compares
random, uncertainty, QBC, and PCPI class-EIG on official CCPP and Gas Turbine
measurements. P3B.2 completed 96/96 real policy runs and passed every protocol
check, but no family-level structural effect or predictive effect established
superiority. CCPP class entropy was already numerically collapsed, and the
acquisition target changed while the reported endpoint stayed fixed. That
artifact is retained as development negative evidence, not a paper
superiority result. P3B.3 fixed the initial class map for the whole sequence.
It uses certified class EIG when available and posterior latent-mean epistemic
variance otherwise. All policies
shared one exact conjugate finite-bank posterior, initial observations,
candidate pool, validation rows, budgets, and seeds. Candidate labels remain
behind `PoolOracle` until the selected index is acquired; untouched-heldout
remains unavailable. The returned P3B.3 run passed all protocol checks but did
not pass the efficacy Gate: CCPP used the epistemic fallback for 98.8% of PCPI
queries and every Gas-family paired efficacy interval crossed zero.

P3B.4 removed the column-order-dependent adjacent-interaction bank and added a
power-likelihood generalized posterior. Its returned 96/96 run passed protocol
but failed efficacy: eta remained 1 for every CCPP seed, CCPP class entropy was
already collapsed, and no family-level primary interval excluded zero.

P3B.5 selected eta by prequential posterior-randomized R-log SafeBayes and
centered/scaled every closed bank term from initial-development covariates.
Its returned archive is internally intact, but the run is invalid efficacy
evidence: posterior fitting used those transformed coordinates while posterior
prediction and acquisition rebuilt raw basis rows. P3B.6 makes the posterior
engine's frozen design transform the only predictive coordinate route. It does
not change formulas, thresholds, seeds, budgets, splits, policies, or
assessment rules.

P3B.6 writes every policy summary, learning-curve point, query decision, and
failure to the single hash-chained `EvidenceRegistry` before creating tables.
CSV, JSON, and figures are read-only exports, with their hashes committed in
`diagnostics/evidence_export_manifest.json`. Per-seed initial, validation, and
candidate subset commitments plus the common candidate-score budget are part
of the protocol Gate.

`hypothesis_mvp.pcpi.real_acquisition` is the only production acquisition
implementation. The earlier candidate-variance planner has been deleted, and
`DiscoveryAgent` fails closed if legacy acquisition is requested. The former
RD1 `no_acquisition` pseudo-ablation is no longer scheduled; P3B owns the
matched-budget acquisition comparison.

## Environment

Use the existing clean Python 3.11 environment:

```powershell
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
& $python -m pip install -e "D:\01\666\hypothesis_mvp[test]"
```

The patch installer does not open held-out, download data, run the real
experiment, alter `.env`, or delete existing outputs.

## Formal P2A.1 correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p2a_1_smc_genealogy_correctness_20260811"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p2a_1_smc_genealogy_20260811.zip"
$config = "$project\configs\p2a_1_correctness_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p2a1_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P2A.1 `
    --config $config `
    --heldout-state not-applicable
```

The frozen 24-run controlled diagnostic passed all Gate decisions. It checks batch/sequential agreement, weight
normalization, adaptive CESS bridges, unbiased-resampling smoke behavior,
ESS-only resampling decisions, complete parent/child/root maps, monotone root
ancestry, invariant rejuvenation, TV/KL convergence, predictive NLL, marginal
likelihood error, and seed stability. It is correctness evidence only.

## Heldout-closed P2A.1 real calibration

```powershell
$project = "D:\01\666\hypothesis_mvp"
$data = "$project\data"
$output = "$project\outputs\p2a_1_robust_smc_uci_mainline_20260807"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p2a_1_robust_smc_20260807.zip"
$config = "$project\configs\p2a_1_robust_smc.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p2a_real `
    --data-root $data `
    --output-dir $output `
    --source-artifact $source `
    --phase P2A.1 `
    --config $config `
    --heldout-state closed
```

The runner recursively finds official files, verifies frozen SHA-256 values,
loads only CCPP `Sheet1`, and groups Gas Turbine by year. Dataset or target names
never choose formula templates. The process prints and flushes run progress,
metrics, failures, and the final Gate decision while appending the same events
to `logs/run.jsonl`.

The result contains `RUN_MANIFEST.json`, one hash-chained
`evidence_registry.jsonl`, every seed and failure, flattened bridge diagnostics,
genealogy telemetry, tables, figures, and an explicit claim boundary. It never
opens untouched-heldout.

## Formal P2B correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p2b_transdimensional_diagnostic_20260811"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p2b_transdimensional_20260811.zip"
$config = "$project\configs\p2b_transdimensional_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p2b_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P2B `
    --config $config `
    --heldout-state not-applicable
```

This 24-run Gate prints every seed as it runs and persists the same events to
`logs/run.jsonl`. It checks exact posterior agreement, particle convergence,
proposal normalization, reverse support, dimension semantics, detailed
balance, target invariance, weights, CESS, genealogy, and all three move types.
It intentionally has no `--data-root`: it is a controlled correctness fixture,
not a real-data performance experiment.

## Tests and audit

```powershell
& $python -B -m pytest -q -p no:cacheprovider
& $python -B scripts\audit_final_source.py
& $python -B -m scripts.run_pcpi_p2a1_diagnostic --help
& $python -B -m scripts.run_pcpi_p2a_real --help
& $python -B -m scripts.run_pcpi_p2b_diagnostic --help
& $python -B -m scripts.run_pcpi_p3a_eig --help
& $python -B -m scripts.run_pcpi_p3b_real --help
& $python -B -m scripts.run_pcpi_p3b3_diagnostic --help
& $python -B -m scripts.run_pcpi_p3b6_predictive_consistency_diagnostic --help
```

## P3A.2 repaired operational-class EIG diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3a_2_gauss_jacobi_class_eig_20260811"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3a_2_gauss_jacobi_class_eig_20260811.zip"
$config = "$project\configs\p3a_exact_class_eig.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3a_eig `
    --output-dir $output `
    --source-artifact $source `
    --phase P3A.2 `
    --config $config `
    --heldout-state not-applicable
```

## P3B.3 controlled decision-rule diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_3_decision_rule_diagnostic_20260811"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_3_decision_aligned_acquisition_20260811.zip"
$config = "$project\configs\p3b_3_decision_rule_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b3_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.3 `
    --config $config `
    --heldout-state not-applicable
```

This controlled fixture checks exact-EIG top-one agreement, both epistemic
fallback branches, and frozen partition identity after a posterior update. It
is correctness evidence only and has no `--data-root`.

## P3B.6 predictive-coordinate correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_6_predictive_consistency_diagnostic_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_6_predictive_consistent_acquisition_20260812.zip"
$config = "$project\configs\p3b_6_predictive_consistency_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b6_predictive_consistency_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.6 `
    --config $config `
    --heldout-state not-applicable
```

This fixture checks the earlier powered-posterior and SafeBayes contracts, then
requires every preconditioned predictive component and epistemic-variance path
to agree with the posterior engine. It also verifies that all four acquisition
policies are invariant to affine unit reparameterization. It is correctness
evidence only.

## P3B.6 predictive-consistent real measured-pool acquisition

```powershell
$project = "D:\01\666\hypothesis_mvp"
$data = "$project\data"
$output = "$project\outputs\p3b_6_predictive_consistent_acquisition_uci_mainline_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_6_predictive_consistent_acquisition_20260812.zip"
$config = "$project\configs\p3b_real_acquisition.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b_real `
    --data-root $data `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.6 `
    --config $config `
    --heldout-state closed
```

P3B.6 prints every policy run and acquisition round while writing the same
crash-durable records to `logs/run.jsonl`. A poor effectiveness result is not
converted into a process failure: it is preserved and classified in
`effectiveness_assessment`. Protocol, hash, budget, or leakage failures return
exit code 2 and make the run invalid.

For PCPI queries, the process records the frozen target-partition hash, utility
mode, raw class-EIG scores and envelopes, evaluation count, and
ranking-certificate state. Nested quadrature starts at 32 evaluations and
doubles up to the frozen 512 cap. A certified leader uses class EIG about the
initial frozen class variable. A singleton class or uncertified capped ranking
uses posterior latent-mean epistemic variance with observation noise excluded.
The fallback is not called EIG.

The returned P3B.6 real run passed its protocol Gate but not its efficacy Gate.
It is retained as valid development evidence and is audited in
`docs/pcpi_p3b6_result_audit_20260812.md`.

## P3B.7 budget-resolved operational-class diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_7_budget_resolved_classes_diagnostic_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_7_budget_resolved_classes_20260812.zip"
$config = "$project\configs\p3b_7_budget_resolved_classes_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b7_budget_resolved_classes_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.7 `
    --config $config `
    --heldout-state not-applicable
```

This correctness fixture verifies the root-budget resolution identity,
partition refinement, normalized class mass, affine-unit invariance, and exact
class-EIG ranking. It has no `--data-root` and is not efficacy evidence.

## P3B.7 returned real measured-pool result

The P3B.7 archive is valid negative development evidence: protocol PASS,
efficacy FAIL, 96/96 runs, zero failures, and held-out closed. Its complete
audit is `docs/pcpi_p3b7_result_audit_20260812.md`. P4/P5 remain blocked.

## P3B.8 joint class--predictive correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_8_joint_class_predictive_eig_diagnostic_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_8_joint_class_predictive_eig_20260812.zip"
$config = "$project\configs\p3b_8_joint_class_predictive_eig_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b8_joint_eig_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.8 `
    --config $config `
    --heldout-state not-applicable
```

This controlled fixture checks the information-chain decomposition,
nonnegative conditional predictive information, the singleton Gaussian
identity, affine-unit and target-order invariance, exact class-EIG agreement,
and certified joint ranking. It has no `--data-root` and is correctness-only.

## P3B.8 returned real measured-pool result

The returned P3B.8 archive is protocol-valid negative development evidence:
96/96 runs, zero failures, held-out closed, and efficacy status
`REAL_ADVANTAGE_NOT_DEMONSTRATED`. CCPP frozen-class gain versus random is
`-0.304986`, 95% CI `[-0.465243, -0.144729]`. The complete audit is
`docs/pcpi_p3b8_result_audit_20260812.md`.

## P3B.9 representative-safe correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_9_representative_safe_joint_diagnostic_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_9_representative_safe_joint_acquisition_20260812.zip"
$config = "$project\configs\p3b_9_representative_safe_joint_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b9_representative_safe_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.9 `
    --config $config `
    --heldout-state not-applicable
```

This fixture reruns all eleven P3B.8 joint-score decisions and adds six
representative checks: MMD update identity, nonempty safe-set behavior,
selected non-increase, exact joint winner inside the safe set, affine/unit and
order invariance, and the explicit minimum-MMD empty-safe-set fallback. It has
no `--data-root` and cannot establish real efficacy.

## P3B.9 returned real measured-pool result

The returned P3B.9 archive is protocol-valid negative development evidence:
96/96 runs, zero failures, held-out closed, nonempty representative safe sets
at every PCPI query, and efficacy status `REAL_ADVANTAGE_NOT_DEMONSTRATED`.
The guard therefore worked as specified but did not repair posterior-score
misspecification. The complete boundary is recorded in
`docs/pcpi_p3b9_result_audit_20260812.md`.

## P3B.10 maximin joint-EIG correctness diagnostic

```powershell
$project = "D:\01\666\hypothesis_mvp"
$output = "$project\outputs\p3b_10_maximin_joint_eig_diagnostic_20260812"
$source = "D:\01\666\hypothesis_mvp_canonical_source_p3b_10_representative_safe_maximin_joint_acquisition_20260812.zip"
$config = "$project\configs\p3b_10_maximin_joint_eig_diagnostic.json"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"

Set-Location $project
& $python -B -m scripts.run_pcpi_p3b10_maximin_joint_eig_diagnostic `
    --output-dir $output `
    --source-artifact $source `
    --phase P3B.10 `
    --config $config `
    --heldout-state not-applicable
```

This correctness-only fixture reruns all seventeen P3B.9 decisions and adds
ten finite-family decisions: exact lower-envelope agreement, interval
containment, safe-set winner agreement, least-favorable-model auditing,
model-order invariance, singleton recovery, and deterministic tie handling.

## P3B.10 returned real measured-pool acquisition

The 27/27 controlled P3B.10 correctness Gate passed. Its subsequent
held-out-closed run completed 96/96 policy runs with zero failures and an
independently valid 97-event EvidenceRegistry, but returned
`REAL_ADVANTAGE_NOT_DEMONSTRATED`. On CCPP, frozen-class gain versus random was
`-0.28378374200721335` with 95% interval
`[-0.5046926179404461, -0.06287486607398057]` and 7/8 negative-transfer seeds.
The grouped Gas-family interval crossed zero. P3B.10 is therefore retained as
protocol-valid negative development evidence, not as pending or positive
efficacy evidence.

## P3C.1 discrepancy-aware real measured-pool acquisition

P3C.1 was the latest real-development candidate. Its controlled implementation
Gate and source-identity audit passed, and the user-executed held-out-closed
run completed with zero failures. The result was protocol-valid but did not
pass the preregistered efficacy Gate; it is retained as negative development
evidence. Do not rerun P3C.1 or promote the mainline without a new
task-independent repair and controlled Gate.

The following command is an **archived identity record**, not an instruction to
rerun the failed candidate. Its output directory is intentionally the existing
audited directory, so the fail-closed runner must refuse to overwrite it.

```powershell
$project = "D:\01\666\hypothesis_mvp"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
$dataRoot = "D:\01\666\data"
$output = "D:\01\666\hypothesis_mvp\outputs\p3c_1_real_discrepancy_d4b24b2_20260813"

Set-Location $project
& $python -B scripts/run_pcpi_p3c_real.py `
    --data-root $dataRoot `
    --output-dir $output `
    --config configs/p3c_1_discrepancy_real_acquisition.json `
    --phase P3C.1 `
    --heldout-state closed
```

This command is an efficacy evaluation, not a guarantee of improvement. A
protocol-valid run with `REAL_ADVANTAGE_NOT_DEMONSTRATED` remains negative
development evidence and does not permit promotion to P4/P5.

## P3D.1 certified reference-dominance diagnostic

P3D.1 is a correctness-only decision repair. It evaluates exact finite
class-EIG and hands over from a registered reference policy only when the
candidate's utility lower bound exceeds the reference policy's utility upper
bound. Overlapping intervals and zero class capacity execute the registered
reference policy; they never switch to posterior variance or another target.

The diagnostic has no real-data, validation, held-out, LLM, motif, or discovery
surface. Passing it does not integrate P3D.1 into
`hypothesis_mvp.pcpi.real_acquisition` and does not authorize another CCPP/Gas
run. The mathematical and evidence contract is in
`docs/pcpi_p3d1_root_cause_and_repair.md`.

```powershell
$project = "D:\01\666\hypothesis_mvp"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
$output = "D:\01\666\hypothesis_mvp\outputs\p3d_1_reference_dominance_correctness"

Set-Location $project
& $python -B scripts/run_pcpi_p3d1_reference_dominance_diagnostic.py `
    --output-dir $output `
    --config configs/p3d_1_reference_dominance_diagnostic.json `
    --phase P3D.1 `
    --heldout-state not-applicable
```

## Current claim boundary

Supported when the frozen P2A.1 Gate passes:

> PCPI implements a Rao--Blackwellized, adaptively tempered fixed-universe SMC
> whose weights, ESS-adaptive resampling, explicit genealogy, invariant
> rejuvenation, and posterior approximation are checked against an exact
> finite-bank reference.

Supported when the frozen P2B and P3A.2 Gates are subsequently revalidated:

> PCPI implements an explicit proposal-corrected collapsed trans-dimensional
> SMC kernel and verifies its finite-bank target invariance and convergence
> against an exact reference posterior.

> PCPI implements uncertainty-scaled complete-link operational predictive-class
> aggregation and a Student-t-measure Gauss--Jacobi class-EIG estimator that
> agrees with independent adaptive quadrature under the P3A.2 correctness
> diagnostic.

The audited P3B.2--P3B.4 results are negative development evidence. P3B.5 is
invalid efficacy evidence because of the coordinate mismatch. P3B.6--P3B.9
are protocol-valid but do not pass the efficacy Gate. P3B.10 and P3C.1 both
have controlled correctness evidence and protocol-valid negative real results;
neither supports acquisition superiority. P3B.10 remains explicitly
generalized Bayes, not an ordinary-likelihood posterior. P3C.1's scalar,
distance-shaped discrepancy envelope is a documented failed repair, not a
contribution claim. None of these fixtures or real-development results supports
open-grammar discovery superiority, motif safety, held-out confirmation, VED
discovery, physical intervention, or a new scientific law.

The clean-source P3D.1 Gate passed all 14/14 frozen decisions at commit
`5d71f588398daac3a7c8d982ec3eac0b5834d73c`; its one-event registry and all
seven exported file hashes verify. It supports only the following statement:

> Given valid simultaneous intervals for the frozen class-EIG utility, the
> implemented handover selects a target-seeking action only when its lower
> bound strictly exceeds the registered reference policy's upper bound;
> otherwise it executes that reference policy.

This is model-relative numerical decision correctness, not a posterior
misspecification repair, real-data efficacy result, or realized no-harm claim.
See `docs/pcpi_p3d1_result_audit_20260814.md` for the frozen identity.
