# PCPI P3H.5 likelihood-power family calibration contract

## Purpose

P3H.5 is the unique family-wide real calibration-only Gate between P3H.4 state
mapping and any future acquisition experiment. It audits the transformed
predictive PIT for every frozen likelihood-power target. It never computes an
acquisition score, opens a candidate response, selects a candidate or accesses
held-out data.

The Gate contains 96 fixed coordinates:

\[
3\ \text{datasets}\times 8\ \text{seeds}\times
4\ \text{likelihood powers}=96,
\]

with \(\eta\in\{0.125,0.25,0.5,1\}\). The power set is inherited from the
robust acquisition ambiguity family and is frozen before P3H.5 results. No
SafeBayes winner or result-dependent subset is used in this Gate.

## Repair of the initial-history leakage surface

P3H.2 standardized the target using all 32 initial responses and then replayed
those same standardized observations through a nominally sequential residual
update. This is acceptable as an initial training construction for its narrow
development diagnostic, but it cannot support the stronger P3H.4 statement
that every raw PIT is based on a strict response prefix: early standardized
targets implicitly depend on later initial responses through the fitted mean
and scale.

P3H.5 removes that ambiguity before new real results by freezing two disjoint
initial roles in the already registered 32-observation budget:

1. the first 16 stored initial observations are the **base warmup**;
2. the final 16 are the **residual-training sequence**.

The response standardizer is fitted only on the warmup responses. The design
preconditioner is fitted only on warmup covariates. Every generalized-Bayes
base posterior first conditions on the complete warmup. Its residual state
remains empty. The remaining 16 observations are then issued one at a time:
the current target-specific base PIT is computed before the response update,
and only that PIT enters the target's KT dyadic state.

This 16/16 split is a role separation fixed without inspecting P3H.5 results.
It is not a tuned efficacy hyperparameter. It preserves the original initial
budget and gives both roles nonempty equal mass. Later work may compare other
preregistered splits only as new experiments; it may not replace this Gate
after seeing its outcome.

## Validation sequence

For one dataset/seed coordinate, validation rows retain the P3H.2 stable
row-ID/SHA-256 order. At validation position \(t\), target \(\eta\) has base
posterior \(P_{\eta,t}\), raw-PIT residual law \(G_{\eta,t}\), and issues

\[
V_{\eta,t}
=G_{\eta,t}\!\left(
F_{\eta,t}(Y_t\mid X_t,\mathcal H_{t-1})
\right).
\]

Only after all four powers issue their corrected PIT for that row does the
runner admit its response to each base posterior and residual state. The
incremental update is tested against complete conditioned reconstruction. Any
changed prefix commitment is terminal.

The 256 validation responses are calibration diagnostics only. The final
validation-updated family is hashed for audit and then discarded. It is never
used to initialize acquisition. A later acquisition run, if authorized, must
reconstruct from base warmup, initial residual-training observations and then
only actually acquired responses.

## Simultaneous error control

P3H.5 retains the fixed P3H.2 PIT basis and betting strategies. The global
false-alarm budget is \(\alpha=1/100\). Equal allocation over all 96 frozen
coordinates gives

\[
\alpha_{d,s,\eta}=\frac{1}{9600},
\qquad
\sum_{d,s,\eta}\alpha_{d,s,\eta}=\frac{1}{100}.
\]

The per-coordinate rejection boundary is therefore 9600. This larger boundary
is the necessary multiplicity correction for adding four likelihood-power
targets; reusing P3H.2's 2400 boundary would spend four times the registered
familywise error budget.

The global compatibility decision is `true` only when all 96 coordinates
complete and none rejects. The terminal states are:

- `FAMILY_CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED` for 96/96
  nonrejections;
- `FAMILY_CALIBRATION_NO_GO` if any coordinate rejects or the complete set is
  not available.

Both states keep acquisition blocked. A compatible result must first be
audited for hashes, completeness and descriptive failure patterns before a
separate operational protocol can be frozen.

## Fail-closed execution order

The unique runner performs, before registered data loading:

1. exact config validation and clean tracked-worktree verification;
2. P3H.1 residual-law correctness evaluation;
3. P3H.3 transformed-utility correctness evaluation;
4. P3H.4 model-specific state-mapping correctness evaluation;
5. P3H.5 deterministic four-power incremental self-check;
6. complete dependency and CPython/DLL/venv binary identity verification.

Only then may it load provenance-verified registered datasets. It evaluates
all coordinates even after a rejection. There is no seed replacement, power
replacement, early success stop, retry, response-dependent role split or
partial acquisition release.

Publication occurs once, after all coordinates, through a new sibling staging
directory. Every file is flushed and fsynced before the staging directory is
atomically renamed to the requested, previously absent output path.
`family_calibration_runs.csv` and `summary.json` are SHA-256 bound by
`manifest.json`. Existing output or staging paths cannot be overwritten.

## Data and leakage boundary

The frozen real datasets, seeds, split seed, initial budget, validation budget
and candidate-pool budget match the P3H development line. Candidate row IDs are
sampled only to create a role commitment proving the same registered pool. The
curator must necessarily parse the registered source table, but its typed
selection view contains acquisition covariates only: no candidate response is
copied or exposed to inference, and the runner never reads a candidate response.
Candidate covariates do not enter calibration and `acquisition_pool.y` cannot
exist on that typed surface. No pool oracle is constructed. Held-out remains
closed.

The runner source contains no acquisition scorer or discrepancy-acquisition
entry point. It emits no query or acquisition-run file. Every calibration row
records `candidate_response_access=false` and `heldout_access=false`; the
summary records `acquisition_executed=false` independently of the result.

## Integrity freeze

P3H.5 requires the exact P3H.2 canonical CPython 3.11.9 environment:

- dependency hash
  `b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`;
- frozen NumPy, SciPy, python-flint and pytest versions;
- base executable, `python311.dll`, venv launcher and `pyvenv.cfg` byte lengths
  and SHA-256 hashes.

The terminal result additionally binds the runner, posterior update,
predictive CDF, residual-family and semiparametric-residual sources. A runtime
or source mismatch fails before data access.

## Claim boundary

Even 96/96 nonrejections establish only compatibility with this fixed betting
family on this registered development audit. They do not prove conditional
calibration, stationarity of raw PIT innovations, correctness of the P3H.3 KL
joint completion, resolution of real acquisition rankings, superiority,
held-out improvement, discovery or paper acceptance.

P3H.5 is not an efficacy experiment. Its result cannot be pooled with a later
acquisition comparison as though it were an independent held-out sample. All
negative coordinates and descriptive diagnostics must remain archived.
