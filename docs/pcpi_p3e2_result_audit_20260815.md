# P3E.2 canonical correctness-run audit — 2026-08-15

Status: **PASS; correctness evidence only; real posterior adequacy and efficacy remain untested**

This record audits the fresh output returned from the canonical P3E.2 runner
after the source-identity and regression checks were completed on the user's
Windows working tree. It is an execution-identity record, not a real-data
experiment report.

## Source and artifact identity

| Item | Recorded value |
|---|---|
| Canonical source commit | `ca4bfce8abefb22c8c1b9c809c3db0801ff2c080` |
| Config SHA-256 | `18767af75669f63ed67670568a30e7c19d5acbc9d72d82b5c9625dedc4f06bf1` |
| Production-code hash | `243a81829eff28088a583ef893f6aae6101df686600c386e7ea202e7154d47b0` |
| Output archive | `p3e_2_posterior_adequacy_correctness.zip` |
| Output archive SHA-256 | `3cee2b87b222ec228b3f0db5770dbae5936cf42cc821a4215026761ae000285b` |
| Fixture timestamp (UTC) | `2026-08-15T07:26:01.895192+00:00` |
| Archive members | `summary.json`, `adequacy_eprocess.csv` |

The archive passes an independent ZIP integrity check. The working tree used
for the run had the expected commit and no uncommitted changes. The full
Python test command completed successfully (234 tests, no failures, skips, or
collection errors).

## Gate result

The summary records `gate_passed=true` for all 11 registered decisions:

- response-free deterministic basis construction;
- orthogonality to the union of candidate designs;
- independent covariance-form marginal checks;
- posterior normalization and coefficient invariance;
- exact-null non-rejection and nominal eligibility;
- structured-residual threshold crossing and reference-only fallback;
- prequential-to-batch Bayes-factor telescoping; and
- unavailable held-out state.

Numerical diagnostics are:

| Diagnostic | Value |
|---|---:|
| Maximum orthogonality error | `4.440892098500626e-16` |
| Independent marginal maximum error | `7.105427357601002e-15` |
| Telescoping maximum error | `0.0` |
| Exact-null log Bayes factor | `-0.713366203484938` |
| Structured-residual first rejection round | `16` |
| Structured-residual log Bayes factor | `4.936581343188919` |

The small difference from other BLAS/NumPy runs in the last displayed
orthogonality digits is within the registered numerical tolerance and does not
change any gate decision.

## Claim boundary

The artifact supports the following narrow statement:

> On the registered finite correctness fixture, the response-free
> union-orthogonal discrepancy construction, exact null/discrepancy marginal
> likelihoods, prequential Bayes-factor e-process, and fail-closed decision
> contract are implemented consistently with the canonical source identity.

It does **not** support real-data posterior adequacy, discrepancy-prior
calibration, efficacy, posterior improvement, held-out performance, motif
transfer, or scientific discovery. The summary explicitly records
`real_data_accessed=false`, `heldout_opened=false`,
`formal_real_posterior_adequacy_evidence=false`, and
`formal_efficacy_evidence=false`.

## Next admissible evidence

P3E.2 correctness is now closed. The next admissible experiment is a separately
frozen, initial-development-only real posterior-adequacy audit using the
official measured data roles and fixed response order. It must report the full
e-process for every registered family/seed and must not compare acquisition
policies or open held-out data. No augmented-posterior acquisition run is
authorized before that audit and a subsequent predictive-calibration Gate.
