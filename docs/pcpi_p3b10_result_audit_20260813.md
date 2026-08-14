# P3B.10 real-result audit - 2026-08-13

Decision: **protocol PASS; real efficacy Gate FAIL; mainline NO-GO**.

This audit records the user-executed frozen real-measurement run. It does not
authorize a paper superiority claim, P4/P5, motif transfer, VED discovery, or
untouched-heldout confirmation.

## Identity and integrity

- Output: `outputs/p3b_10_real_maximin_joint_acquisition_f9be2491_20260813`.
- Source Git commit: `f9be2491633b5e1653758a9b5751ab99ea919e6d`.
- Source Git tree: `1f54a3d85df23c8d94413ef2554af3151f2cfcf4`.
- Tracked-source tree digest: `69a30d737a825ca0e4e3472476449e10c04aa56d8b398a4782f2d559f3d54543`.
- Production-code hash: `9482cabc848c5aa1aaf60a695547da089b13831cb3962ac12467904a3c233a3b`.
- Configuration hash: `842fdaa7cb7768492bd7f69d2e52dc5ca6f6308e5ff7273ee23bb759bb9e7011`.
- Exact dependency-environment hash:
  `d8eb9c5b56333fcb3a964b7f9d18b21801773378ecfb479ae559a80edca464ad`.
- Source was clean (`source_git_dirty=false`); official UCI source hashes were
  verified; EvidenceRegistry has 97 valid events with head
  `9674a90a4803d53e273c82e2f54cc8694a50f68908af8b9fb8911d63dd3435b0`.

Independent output checks found no exported-file hash mismatch, and the stored
row counts are exact: 96 policy runs, 3,168 learning-curve rows, 3,072 query
records, 12 aggregates, and 15 paired effects. There were zero failed runs.

## Frozen protocol boundary

The run used the registered measured UCI CCPP and Gas Turbine targets, four
matched-budget policies, eight frozen seeds, 32 initial observations, 32
acquisitions, 128 candidate-pool observations, and 256 validation observations.
The initial class partition, SafeBayes calibration, design preconditioner,
predictive target domain, ambiguity set, and representative guard were shared
as preregistered. `heldout_opened=false` and `selection_used_heldout=false`.

Protocol decisions all passed, including official hashes, complete runs and
curves, matched budgets, shared subset commitments, auditable representative
decisions, auditable maximin decisions, and closed held-out state.

## Efficacy result

The stored assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED`:

- CCPP vs random frozen-class gain: mean paired delta
  `-0.28378374200721335`, 95% paired t interval
  `[-0.5046926179404461, -0.06287486607398057]`, negative transfer in 7/8
  seeds.
- Grouped Gas Turbine vs random frozen-class gain: mean paired delta
  `0.034959682233189106`, 95% interval
  `[-0.2603092503809608, 0.3302286148473398]`.
- `pcpi_decision_rule_valid=true`, but
  `strong_evidence=false` and `strong_structural_evidence=false`.
- PCPI predictive nAULC is not non-positive versus every baseline in every
  family, although it is better than random in the preregistered family-level
  aggregate. This does not rescue the primary class-discrimination claim.

The maximin repair was actually active: relative to P3B.9, same-round PCPI
choices overlap 48.8% on CCPP, 17.2% on Gas CO, and 12.5% on Gas NOX. The
negative CCPP result therefore cannot be attributed to a no-op implementation
or a protocol/package mismatch.

## Methodological interpretation

This result is evidence against the current finite polynomial-bank posterior
and its joint class/predictive utility as a generally reliable real-data
acquisition rule. Tempering the same six closed polynomial structures changes
posterior concentration but does not represent model discrepancy, omitted
variables, heteroskedastic noise, or regime structure. The finite lower envelope
therefore cannot be treated as a universal misspecification guarantee.

The next repair must be general and pre-specified. Candidate directions are a
predictive model-discrepancy component or a formally justified robust decision
utility that preserves the primary class target, with validation and real-data
efficacy locked only after a new controlled correctness Gate. No dataset-specific
formula, target-direction branch, result-derived threshold, seed, or regularizer
may be introduced.
