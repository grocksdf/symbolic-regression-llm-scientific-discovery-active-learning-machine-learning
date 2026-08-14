# P3B.10 local-source audit - 2026-08-13

Decision: **PASS to execute the frozen real-measurement P3B.10 protocol once.**
This is not a P3 efficacy pass and does not unblock P4, P5, held-out
confirmation, or submission claims.

## Evidence audited

- Audited algorithm base: Git commit `283bc9ac4cdf2b8dc0a33ff55f635af9d3d33e04`.
- The returned focused command completed 69/69 tests in the canonical project
  environment: `test_pcpi_p3_acquisition.py`, `test_integrity.py`, and
  `test_pcpi_leakage_boundaries.py`.
- `test_maximin_joint_diagnostic_passes_twenty_seven_decisions` executes the
  controlled evaluator and requires `gate_passed`, exactly 27 decisions, every
  decision true, and equality of the estimated and exact selected action.
- The final static source audit returned zero failures.
- No acquisition or posterior algorithm changed after the returned 27/27
  result. Commit `fbc45e65` changes source-evidence identity only.

## Archive-free source identity

Formal P3B.10 runners may now omit `--source-artifact`. In that mode they fail
closed unless the runner is inside the Git worktree root and Git reports no
staged, unstaged, or untracked files. The run manifest records:

- Git commit and Git tree;
- SHA-256 over tracked paths, modes, and local file bytes (with pinned gitlinks);
- production Python code SHA-256;
- configuration content and file hashes;
- dependency-specification and exact runtime-environment hashes.

The prior verified-ZIP mode remains available for backward compatibility.

## Frozen real-run boundary

The allowed run uses only the registered UCI CCPP and Gas Turbine measurement
files, eight frozen seeds, four matched-budget policies, and mandatory official
source hashes. Selection receives development responses, validation responses,
acquisition-pool covariates, and queried pool labels through `PoolOracle`; it
has no held-out capability. Untouched-heldout remains commitment-only and
closed. Any missing run, substituted seed, hash mismatch, protocol failure, or
failed efficacy rule is a `NO-GO` and must not be repaired by data-specific
thresholds or result-driven tuning.
