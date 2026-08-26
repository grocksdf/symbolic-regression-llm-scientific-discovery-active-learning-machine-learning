# PCPI P3H.7 semiparametric real acquisition protocol

## Purpose

P3H.7 is the first real measured-pool acquisition protocol allowed to consume
the P3H semiparametric residual family. It is a new frozen development
comparison, not a reinterpretation or rerun of P3B/P3C. The user alone executes
it once after source, configuration and canonical-runtime identities are
frozen.

## Matched design

The protocol retains the three registered real targets, eight registered seeds,
split seed `20260807`, 32 initial measurements, 32 acquired measurements, 128
candidate rows and 256 validation rows. It compares random, uncertainty, QBC
and PCPI under identical row commitments and measured-response budgets.

All policies use the same response standardizer and design preconditioner fitted
only on the first 16 initial observations. All use the eta-one posterior for
reporting and validation. No SafeBayes selection is performed. For PCPI, the
last 16 initial observations form the strict-prefix residual-training history
for the complete frozen powers `(0.125, 0.25, 0.5, 1.0)`.

## PCPI decision path

At every acquisition round, PCPI:

1. reconstructs no validation state and receives no candidate target;
2. scores visible candidate covariates using the P3H transformed class utility
   bound to each power's exact posterior and residual law;
3. takes the minimum joint utility across all four powers;
4. requires the representative safe set and transformed numerical ranking to
   be resolved;
5. freezes the chosen global candidate ID, transformed action and current
   family hash;
6. asks the pool oracle for only that chosen response;
7. verifies the returned ID/action and advances every family target once.

An unresolved ranking, empty representative safe set, legacy fallback, changed
history, wrong oracle response or stale decision fails that run. It does not
switch to Student-t EIG, posterior variance or minimum-MMD selection. Every
query records the family hash before and after update; adjacent hashes must form
one uninterrupted chain.

## Frozen environment and result handling

The runner requires dependency identity
`b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`
before creating the output directory or loading registered data. Official data
hash verification is mandatory. A clean Git source identity, configuration
hash, package snapshot, source/data/split identities, budgets, failures and
held-out flags are recorded in the existing evidence registry and manifest.

All 96 policy runs must complete: three targets times eight seeds times four
policies. A PCPI numerical abstention is a protocol failure and remains in the
failure ledger; no seed, candidate, power or policy run is retried or replaced.

## Assessment and claim boundary

Protocol validity requires all matched runs, curves and queries; shared
subsets, class partition, eta-one reporting target and preconditioner; complete
four-power PCPI decisions; continuous family hash chains; no P3H state on
baselines; closed held-out; and a 100% valid PCPI decision rate.

Efficacy is assessed only by the already frozen paired rules: normalized
validation-RMSE AULC, frozen-class entropy gain, negative-transfer control and
dataset-family aggregation, with CO and NOx counted as one Gas family. Negative,
mixed, invalid and failed outcomes are retained.

Even a strong result is real-development evidence only. It cannot establish
universal superiority, untouched-held-out confirmation, physical intervention,
open-grammar discovery, a new scientific law, AISTATS acceptance or any claim
outside the registered finite-bank measured-pool task.
