# P3F.4-CERT.14 certified function-space resident common target

Status: **RESPONSE-FREE DEVELOPMENT GATE PASSES; ALL OPERATIONAL EXECUTION REMAINS BLOCKED**

GitHub baseline: `1f8a6ff50d62d4ebb3a3e7db5f953a7ee0299127`

CERT.14 closes the target mismatch left explicit by CERT.13.  The resident
bridge and local/RJ source previously evaluated a collapsed likelihood through
a floating factor basis and NumPy linear algebra, while the operational
predictive provider used the 512-bit Arb factorisation-free function-space
covariance.  Those two computations could not be assumed to define one target.

This phase replaces the resident floating target authority with a common
certified composition.  It does not run SMC, draw a particle, materialize an
island, inspect real or held-out data, or expose an operational result.

## 1. One response-free prior constructor

For semantic polynomial design `g`, RBF matrix `K`, coefficient precision
`lambda_beta` and discrepancy precision `lambda_delta`, the shared prior is

\[
K_\perp=K-Kg(g^T K g)^{-1}g^T K,
\]

\[
P=\lambda_\beta^{-1}gg^T+\lambda_\delta^{-1}K_\perp,
\qquad m=m_\beta g.
\]

The identically zero polynomial retains the exact vacuous constraint
`K_perp=K`.  The same private Arb builder now supplies both:

1. the CERT.13 full-`H0` posterior-predictive parameter balls; and
2. the CERT.14 weighted collapsed log-marginal balls.

The builder reads the exact registered action domain, polynomial key, component
state and frozen prior parameters.  It does not read any response.  Therefore
the discrepancy covariance, standardizer and semantic state are identical in
the predictive, bridge-weight and local/RJ paths.

The projected covariance is the Gaussian conditional covariance under the
linear orthogonality constraint.  The orthogonal-GP motivation is documented
by Plumlee and Joseph, *Orthogonal Gaussian Process Models*, Statistica Sinica
28 (2018):

- https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html

## 2. Exact rational bridge powers

At observation index `t`, previous observations have power one and the current
observation has exact registered power

\[
\beta=b/B,
\]

where `b` and `B` are integers from the frozen Feynman--Kac grid.  Future
responses have weight zero and are not accessed.

Let `W` be the diagonal matrix of included likelihood powers, let
`D=sqrt(W)`, and let `r=y-m`.  CERT.14 evaluates the symmetric system

\[
C_W=I+DPD.
\]

The weighted Gaussian/NIG collapsed log marginal is

\[
\log p_W(y\mid T,d)
=-\frac{n_W}{2}\log(2\pi)
-\frac12\log\det C_W
+a_0\log b_0-a_W\log b_W
+\log\Gamma(a_W)-\log\Gamma(a_0),
\]

where

\[
n_W=\operatorname{tr}(W),\qquad
a_W=a_0+\frac{n_W}{2},
\]

\[
b_W=b_0+\frac12(Dr)^TC_W^{-1}(Dr).
\]

This is the function-space determinant/quadratic identity equivalent to
integrating the registered coefficient, discrepancy and NIG noise state.  It
is not a generalized final posterior: fractional power exists only along the
registered intermediate Feynman--Kac bridge and the terminal target uses
`beta=1`.

The Gate independently checks the inactive-component terminal ball against
the conjugate parameter-space precision formula.  The higher-precision result
is contained in the 512-bit function-space ball.

Gaussian-process conditioning and log-marginal identities are given in
Rasmussen and Williams, *Gaussian Processes for Machine Learning*, Chapter 2:

- https://gaussianprocess.org/gpml/chapters/RW2.pdf

## 3. Validated numerical contract

All RBF, square-root weight, determinant, logarithm, log-gamma, quadratic and
linear-solve operations use `python-flint==0.8.0` at one registered 512-bit
precision.  The only solve is

```text
arb_mat.solve(..., algorithm="precond")
```

The determinant, posterior shape and posterior scale must have strictly
positive lower endpoints.  An indeterminate or invalid ball raises an error.

The following are forbidden:

- `np.linalg.solve`, `slogdet`, an explicit inverse or `algorithm="approx"`;
- floating `eigh`/SVD factor bases or tolerance-selected rank;
- jitter, ridge, diagonal loading or another regularizer;
- `nextafter`, midpoint promotion or rounded resident snapshots;
- result-dependent precision retry or a fallback solve.

python-flint documents that `arb_mat.solve` with `precond` retains error bounds
and that `algorithm="approx"` does not:

- https://python-flint.readthedocs.io/en/latest/arb_mat.html

Arb's outward ball-arithmetic model is described by Johansson, *Arb: Efficient
Arbitrary-Precision Midpoint-Radius Interval Arithmetic*:

- https://arxiv.org/abs/1611.02831

## 4. Bridge potential identity

For one state and adjacent bridge powers, CERT.14 constructs

\[
G_{t,b\rightarrow b'}(T,d)
=\frac{p_{W_{b'}}(y\mid T,d)}{p_{W_b}(y\mid T,d)}.
\]

The exported log-potential ball is the outward interval difference

\[
[L_{b'}-U_b,\;U_{b'}-L_b].
\]

Both endpoints carry the same common-plan, provider, semantic state,
observation, history and beta-grid identities.  Cross-state, cross-history,
cross-observation or non-increasing beta endpoints fail closed.

## 5. Exact local/RJ composition

For the CERT.5 involutive proposal edge `x -> x'`, CERT.14 uses:

- exact raw-AST prior fractions;
- exact component prior fractions;
- exact forward and reverse auxiliary proposal fractions;
- the two certified collapsed log-marginal balls at the identical bridge
  coordinate; and
- the already proved discrete unit Jacobian.

The MH log-ratio ball encloses

\[
\log\frac{p_{W}(y\mid x')p(x')q(x\mid x')}
               {p_{W}(y\mid x )p(x )q(x'\mid x )}.
\]

The acceptance ball is the outward image of `min(0, log-ratio)`.  If the ratio
straddles zero, the result remains an interval ending at zero.  CERT.14 does
not select a favourable endpoint or silently turn an unresolved comparison
into an accept/reject decision.

A finite rational transition audit separately constructs the MH matrix from
positive target masses and a stochastic proposal matrix.  It verifies every
pairwise detailed-balance identity and the complete target-invariance identity
exactly with `Fraction`; this is deterministic combinatorics, not a simulation.

## 6. Sparse fixed-candidate adapter

The common plan binds:

- the CERT.13 provider hash;
- the frozen `H0`, standardizer and domain hashes;
- the CERT.8 Feynman--Kac plan hash;
- the CERT.7 local/RJ composition and CERT.5 proposal-plan hashes;
- the CERT.12 Arb CDF kernel hash; and
- the CERT.13 sparse fixed-candidate projector hash.

The response-free adapter accepts only exact finite fixture state masses and
the selection-fixed candidate class.  It returns the same lower/upper candidate
mass bounds as CERT.13 while retaining the CERT.14 target identity.  It never
materializes `6^d`, appends `other`, chooses a nearest bin, or normalizes a
componentwise median.

Operational sparse-result access remains guarded before result, particle,
state or candidate inspection.

## 7. Resident source retirement boundary

`ScalableOpenTargetSMC._bridge_log_marginals` now rejects the resident
raw-state branch before the historical floating factor-basis collapsed target
can run.  `_rejuvenate` preserves the earlier unbound-target guard, then rejects
valid resident execution before proposal sampling.  `run()` retains its first
guard before `_validated_data`, response-energy workspace construction or
particle sampling.

The old floating design implementation remains available only to historical
nonresident correctness paths.  It is not an authorized resident target and is
not combined with CERT.13 results.

## 8. Authorization boundary

The only new authorization is:

```text
P3F4_CERT14_STANDALONE_COMMON_TARGET_COMPOSITION_AUTHORIZED = true
```

All operational switches remain false:

```text
P3F4_CERT14_FLOAT_FACTOR_BASIS_RESIDENT_TARGET_AUTHORIZED = false
P3F4_CERT14_OPERATIONAL_TARGET_RESULT_ACCESS_AUTHORIZED = false
P3F4_CERT14_OPERATIONAL_SPARSE_RESULT_ACCESS_AUTHORIZED = false
P3F4_CERT14_ISLAND_EXECUTION_AUTHORIZED = false
P3F4_CERT14_RESIDENT_SMC_INTEGRATION_AUTHORIZED = false
P3F4_CERT14_RESIDENT_SMC_RUN_AUTHORIZED = false
```

All CERT.10--CERT.13 entropy, product-source, island, CDF/projector-result and
SMC switches also remain false.

## 9. Response-free Gate

CERT.14 retains the 109 CERT.3--CERT.13 checks and adds 14 checks covering:

1. the authorization boundary;
2. complete common-plan identity binding;
3. reuse of one prior-covariance builder by predictive and collapsed targets;
4. the exact zero-power prior target;
5. independent parameter/function-space terminal identity;
6. outward bridge-potential composition;
7. prefix isolation from a changed future response;
8. exact local/RJ forward/reverse ratios;
9. cross-beta and crossed-endpoint rejection;
10. exact finite MH detailed balance and invariance;
11. sparse fixed-candidate target binding;
12. source retirement and pre-data resident guard;
13. absence of inverse, retry, regularization and floating factor bases; and
14. the pre-access operational result guard.

The registered identity is `123/123`; 229 Python files are syntax checked.
No simulated, formal, real, held-out or confirmatory experiment is run.

## 10. Remaining execution gap

CERT.14 proves the common mathematical target but deliberately does not convert
target balls into sampled indices or accept/reject decisions.  Taking interval
midpoints would discard the proof.

The next admissible phase is a response-free certified comparison/sampling
Gate.  It must prove outward log-normalization, exact-bit random thresholds,
multinomial inverse-CDF resampling and MH uniform comparisons with unresolved
comparisons propagated as complete failures.  Any precision schedule and bit
budget must be frozen independently of observed performance.  Only after that
Gate may a separate integration phase consider resident or island execution.

Until then, entropy capture, product-stream materialization, island execution,
resident SMC, predictive calibration, real data, acquisition, held-out access,
confirmation, efficacy and paper-superiority claims remain blocked.
