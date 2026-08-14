# P3C.1 real-result audit — 2026-08-13

Decision: **protocol PASS; real efficacy Gate FAIL; mainline NO-GO**.

This audit records the user-executed P3C.1 run on the registered real UCI
measurements. It does not authorize a superiority claim, P4/P5, untouched-
heldout confirmation, motif transfer, VED discovery, physical intervention,
or a new scientific-law claim.

## Identity and integrity

- Output: `outputs/p3c_1_real_discrepancy_d4b24b2_20260813`.
- Source Git commit: `d4b24b2075926db4fd9ca3cefc5637a7c2378d13`.
- Source Git tree: `0ba40f1f2b7a45cc87a7ca35db4aec3355054b75`.
- Tracked-source tree digest: `5b67eccfc4c90107649d5418c279f169751fa4f7bc4c203f7921b3e70a7b0f51`.
- Production-code hash: `57f0428b67b5d8b5ef4a70fc04f5dd057135f6923614982592c7f69d4ce518f8`.
- Configuration hash: `1b98f4ae36e4e79c52166c1c7a9c7cd3d67e897c2d6cb08c4cd8c47576772717`.
- Configuration-file hash: `dd4e1ff725e2def5d904048b4614f2093122bfae7d3acb816c2ca892e5b0d0b7`.
- Dependency-environment hash: `d8eb9c5b56333fcb3a964b7f9d18b21801773378ecfb479ae559a80edca464ad`.
- Source was clean (`source_git_dirty=false`), official UCI hashes were
  verified, and the 97-event EvidenceRegistry independently verified as valid.
- All nine exported evidence files matched their recorded SHA-256 hashes.

The output contains 96 policy runs, 3,168 learning-curve rows, 3,072 query
records, 12 per-dataset aggregates, and 15 paired effects. There were zero
failed runs. `heldout_opened=false` and `selection_used_heldout=false`.

## Frozen protocol

The run used official measured CCPP and Gas Turbine data, eight registered
seeds, 32 initial observations, 32 acquisitions, 128 candidate-pool rows, 256
validation rows, the four frozen likelihood powers, the initial-frozen class
partition, the covariate-only representative guard, and a closed held-out
state. Every registered protocol decision passed, including hash verification,
matched budgets, shared subset commitments, auditable discrepancy values,
closed-loop oracle isolation, and baseline isolation.

## Efficacy result

The stored assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED` with
`formal_protocol_evidence=true` and `formal_efficacy_evidence=false`.

- **CCPP:** PCPI versus random frozen-class gain had mean paired delta
  `-0.28378374200721335`, 95% interval
  `[-0.5046926179404461, -0.06287486607398057]`; 7/8 seeds were negative
  transfer. Predictive normalized AULC delta versus random was
  `-0.01786905910337984`, with interval
  `[-0.06093553313665106, 0.02519741492989138]`.
- **Grouped Gas Turbine:** PCPI versus random frozen-class gain had mean delta
  `0.06591138158268409`, 95% interval
  `[-0.22906202511823345, 0.3608847882836016]`. Predictive normalized AULC
  delta was `-0.01661885525619368`, interval
  `[-0.10675696359050248, 0.07351925307811512]`.
- Against uncertainty and QBC, the family-level predictive deltas were
  positive (`0.01661432306359162` and `0.00825549404367891`, respectively),
  where positive means worse for PCPI because lower RMSE is better.
- `strong_evidence=false` and `strong_structural_evidence=false`.

The discrepancy repair was not silently disabled. All 768 PCPI query records
used the frozen discrepancy method. It was effectively zero on CCPP (maximum
residual excess variance `6.88e-13`), while it was active on Gas CO and NOX:
mean selected discrepancy variance was `0.3281037555476346` and
`0.5800944857531327`, respectively. Activation on the Gas family did not
produce a statistically decisive or family-uniform advantage.

## Interpretation and next gate

This is evidence against the current discrepancy envelope as a generally
reliable acquisition repair, not evidence of a protocol or implementation
failure. The current profile adds a scalar, distance-shaped variance envelope
to the same finite polynomial-bank posterior. It does not represent
outcome-specific heteroskedasticity, latent operating regimes, omitted
covariates, or cross-target structural dependence. On CCPP its preregistered
residual-excess statistic is essentially zero; on Gas it changes the score but
does not improve the primary class-discrimination claim.

The mainline therefore remains in P3. The next repair must be a new,
task-independent statistical model with a controlled correctness Gate before
any further real-data run. It must be frozen independently of these outcomes,
use only development-role information for calibration, preserve the primary
class target and baseline protocol, and avoid dataset-name branches, result-
derived thresholds, formula-specific rules, or extra seeds/budgets.

