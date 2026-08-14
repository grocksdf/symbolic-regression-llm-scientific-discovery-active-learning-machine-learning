# P3D.2 real reference-dominance protocol audit — 2026-08-14

Decision: **IMPLEMENTATION AND CORRECTNESS GATES PASSED; REAL EXECUTION PENDING**

## Evidence identity

- Stage: `P3D.2`.
- Experiment:
  `real_measurement_matched_budget_reference_dominance_class_eig`.
- Hypothesis identity: `pcpi-p3d2-real-reference-dominance-class-eig`.
- Implementation commit: `cd05e1a`.
- Canonical-data restoration: upstream `81f7cde`, merged at `bb69853`.
- Frozen config:
  `configs/p3d_2_reference_dominance_real_acquisition.json`.
- Runner: `scripts/run_pcpi_p3d_real.py`.
- Real P3D.2 registry, manifest, and result identity: **not yet created**.

The implementation commit is a source identity, not a result identity. No raw
CCPP or Gas Turbine data were present or synthesized during this Gate, no real
policy run was executed, and held-out data were not opened.

## Method contract

The primary utility is only

\[
U_t(a)=I(C_0;Y_a\mid H_t),
\]

where `C0` is the initial-frozen operational class. P3D.2 registers the uniform
distribution over currently visible candidate identifiers as the response-free
reference policy. A stable SHA-256 seed determines the reference draw.

For each action, deterministic response quantization supplies a
data-processing lower bound. Gaussian maximum entropy and within-class
mixture-entropy concavity supply an upper bound. The quantization levels and
`1e-10` numerical outward tolerance are frozen in the config. A targeted
handover occurs only when the largest candidate lower bound strictly exceeds
the reference-probability-weighted upper bound plus the decision roundoff
tolerance. Otherwise the selected candidate is exactly the registered
reference draw.

The real PCPI branch does not apply:

- class-conditional EPIG or a joint class-predictive score;
- representative MMD;
- a finite likelihood-power maximin ranking;
- discrepancy variance inflation; or
- posterior-epistemic fallback.

The likelihood-power candidates remain only in the common
initial-development SafeBayes calibration shared by every policy. They are not
an acquisition ambiguity set in P3D.2.

## Correctness evidence

The P3D.2 tests verify:

1. analytic lower and upper bounds contain an independently integrated
   continuous Student-t-mixture class-EIG correctness fixture;
2. both bounds lie in `[0,H(C0)]`;
3. positive affine response changes preserve the bounds;
4. one-class capacity yields exactly zero lower and upper bounds;
5. malformed quantization contracts fail closed;
6. scoring receives posterior state, candidate covariates, identifiers, and a
   frozen partition, but no candidate, validation, or held-out response;
7. candidate permutations preserve the targeted and reference identities;
8. unresolved intervals return exactly to the reference policy;
9. the real config excludes the P3B/P3C acquisition-module contracts;
10. the CLI requires a local real-data root and closed held-out state; and
11. a three-round `inference_correctness_diagnostic_fixture` loop records
    reference decisions, reveals only selected fixture observations, and
    applies none of the excluded acquisition modules.

Validation at the implementation Gate:

- new P3D.2 test file: 13 tests;
- complete repository suite: 199 passed, zero failures, skips, or collection
  errors;
- Python syntax inventory: 123 tracked Python files;
- production static audit: 63 Python files, 15,355 lines, zero failures.

The Student-t CDF and entropy implementations use SciPy floating-point special
functions. The information inequalities are analytic in exact arithmetic, but
the software is not verified interval arithmetic. The resulting reference
dominance statement remains conditional on numerical containment and on the
finite-bank posterior. It is not a posterior-misspecification repair or a
real-world no-harm guarantee.

## Frozen real protocol

- measured datasets: UCI CCPP, Gas Turbine CO, and Gas Turbine NOX;
- dataset families for inference: CCPP and grouped Gas Turbine;
- seeds: eight frozen seeds `2026080701`--`2026080708`;
- policies: random, uncertainty, QBC, and P3D.2 reference dominance;
- budgets per run: 32 initial, 32 acquired, 128 candidate-pool, and 256
  validation observations;
- expected policy runs: 96;
- expected candidate evaluations per successful policy run: 3,600;
- official-source hash verification: mandatory;
- failure handling: fail closed, record every failure, replace no seed;
- held-out state: closed.

Protocol validity and positive efficacy are exported separately. A complete
96/96 run can still validly return `REAL_ADVANTAGE_NOT_DEMONSTRATED`.

## Local Windows command

```powershell
$project = "D:\01\666\hypothesis_mvp"
$dataRoot = "D:\01\666\data"
$python = "D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe"
$output = "D:\01\666\hypothesis_mvp\outputs\p3d_2_reference_dominance_real"

Set-Location $project
& $python -B scripts/run_pcpi_p3d_real.py `
    --data-root $dataRoot `
    --output-dir $output `
    --config configs/p3d_2_reference_dominance_real_acquisition.json `
    --phase P3D.2 `
    --heldout-state closed
```

The output directory must be new or empty. After completion, return the whole
output directory without editing its JSON, JSONL, or CSV files. The result
audit must verify the EvidenceRegistry chain, every evidence-export hash,
source/config/dependency identity, all 96 run records, paired seed counts, and
the registered assessment before changing any manuscript efficacy wording.
