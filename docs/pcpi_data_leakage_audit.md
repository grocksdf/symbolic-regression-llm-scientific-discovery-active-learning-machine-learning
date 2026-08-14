# PCPI data leakage and hard-coding audit

Stage: P2A real-run candidate  
Verdict: **real selection boundary implemented; held-out remains closed**

## Enforced controls

- Only allowlisted UCI dataset IDs and official SHA-256 values are accepted.
- CCPP loads `Sheet1` once; Gas Turbine loads 2011–2015 and preserves year groups.
- CCPP role assignment hashes row IDs; Gas Turbine roles are frozen by year.
  Neither rule reads targets.
- `SelectionData` exposes development, validation, and acquisition-pool
  covariates only. Its role manifest uses a covariate-only pool fingerprint.
- No held-out array, path, shape, range, summary, target, or descriptive
  metadata crosses into the inference call.
- The generic finite bank is a pure function of feature dimension. Dataset ID,
  filename, family name, target name, and known scientific formulas do not
  choose structures.
- Oracle-expression fields remain blocked by static and runtime tests.
- The real runner has no hash-bypass or held-out-open option.
- Evidence records `heldout_opened=false` and `selection_used_heldout=false`;
  the split manifest stores only an untouched row-ID commitment.

## Remaining boundary work

The local orchestration process necessarily loads official raw files before it
constructs the capability-restricted selection object. A separate-process
confirmation capability and one-shot immutable confirmation ledger are still
required at P6. They are not needed to run this closed-held-out P2A replay, but
P6 cannot pass without them.

Memory remains excluded from P2A. NIST frozen families, Airfoil untouched
held-out, and all VED data are not accessed by the runner.
