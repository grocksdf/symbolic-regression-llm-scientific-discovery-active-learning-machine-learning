# P3G.5 real-result audit

## Identity and integrity

The user-executed archive binds source commit
`86e37abb5ef5ff13c3cf4e151df60cee90c23df2`, the registered P3G.5 config,
the canonical Python 3.11.9 dependency snapshot, and the provenance-verified
UCI CCPP and five annual Gas source hashes. The manifest hashes for
`calibration_runs.csv` and `summary.json` both verify.

The run used measured data only. `simulated_experiment=false`,
`heldout_opened=false`, and candidate responses remained sealed.

## Frozen result

All 24 prequential audits completed. Eight crossed the unchanged `2400`
boundary:

- CCPP: seed `2026080706`;
- Gas CO: seeds `2026080702`, `2026080703`, `2026080704`, `2026080705`,
  `2026080706`, and `2026080708`;
- Gas NOx: seed `2026080705`.

The terminal status is `CALIBRATION_NO_GO`. Acquisition count is zero and no
efficacy claim is supported.

## Interpretation

Relative to P3G.4, the observed linear year nuisance reduced the Gas CO
geometric-mean maximum-e-value ratio to approximately `0.49`. It resolved Gas
CO seeds `2026080701` and `2026080707` and Gas NOx seed `2026080702`, but Gas
CO seeds `2026080704` and `2026080706` and Gas NOx seed `2026080705` newly
crossed. The total rejection count stayed at eight.

Thus year is a real but incomplete nuisance. The persistent central PIT
concentration shows that the finite conditional response family remains
misspecified. P3G.5 rejects further fixed-prior, finite-state noise, or linear
regime patches as the immediate mainline. A future repair must replace the
residual law itself, preserve prequential response isolation, and pass a new
correctness gate before another real execution.
