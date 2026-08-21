# P3F.4-CERT.13 full-H0 parameter balls and sparse candidate projector

Status: **RESPONSE-FREE DEVELOPMENT GATE PASSES; OPERATIONAL ACCESS AND ALL EXECUTION REMAIN BLOCKED**

GitHub baseline: `ef316ea7a7e2b61d796eb05f1b47b89fadfe2ac8`

CERT.13 implements the source and rigorous-numerics layer left open by
CERT.12. It reconstructs a complete vector of Student-t posterior-predictive
parameter balls directly from a frozen exact `H0` identity for any semantic
polynomial/component state, and it composes those intervals with a
candidate-only sparse confirmation projector.

The Gate uses only exact rational algebra, pinned Arb operations, analytic
identities, source inspection and finite deterministic combinatorics. It does
not read a resident particle, capture entropy, run an island, invoke resident
SMC, access real or held-out data, or materialize a formal experiment.

## 1. Root defect resolved by CERT.13

The resident numerical path stores rounded NumPy arrays for the design,
posterior mean/covariance, noise shape and noise scale. Its active-discrepancy
design is produced by floating `eigh` and SVD operations with numerical-rank
thresholds. Wrapping those outputs in zero-radius balls would certify only one
rounded computation. Adding `nextafter`, jitter or a tolerance would not prove
containment of the frozen mathematical posterior.

CERT.13 therefore reconstructs the mathematical target from its frozen source
objects. It never consumes `OpenTargetParticleSnapshot` or
`ScalableOpenTargetResult` in the pure provider:

1. exact binary identities for `H0` actions and responses;
2. exact binary identities for the registered action and threshold grids;
3. the exact semantic polynomial key and component state;
4. exact binary identities for every frozen prior and kernel parameter; and
5. the frozen target, history, standardizer and operational-estimand hashes.

The construction is not a correction selected from observed performance. It
is a response-independent reformulation of the already registered Gaussian,
normal-inverse-gamma and structure-wise discrepancy model.

## 2. Exact `H0` and standardizer identity

Every finite Python input value is identified by `float.as_integer_ratio()`.
This records the exact stored binary value rather than reinterpreting it as a
shorter decimal.

The registered RBF domain is the sorted unique union of:

- actions occurring in frozen `H0`; and
- the frozen selection-visible action grid `A0`.

The standardizer uses only that response-independent domain. A coordinate is
active if and only if its exact action values are not all equal. Its center is
the exact arithmetic mean and its scale is the Arb square root of the exact
population variance. There is no machine-epsilon threshold, observed-response
selection or result-derived rank.

The standardizer hash records the complete exact domain, active columns,
centering/scaling formulas and the no-response rule. The provider refuses a
history or standardizer hash mismatch.

## 3. Exact semantic design on the countably open support

For a polynomial key

\[
  g(x)=\sum_{(p,c)} c\prod_j x_j^{p_j},
\]

all coefficients and powers are integers and all registered action values are
dyadic rationals. CERT.13 evaluates `g` using `fractions.Fraction`; neither a
raw AST identity nor a floating expression evaluation enters the provider.
Every raw AST in the same semantic class therefore produces the same design.

The identically zero polynomial is part of the full open support. For that
state the constraint `g^T delta=0` is vacuous, so the projected discrepancy
covariance is exactly the unprojected RBF covariance. CERT.13 does not divide
by a zero Gram scalar or delete this state.

## 4. Factorisation-free structure-wise projected RBF

On the standardized finite domain, let

\[
  K_{ij}=\exp\left(-\frac{\|z_i-z_j\|^2}{2\ell^2}\right).
\]

For nonzero structure design `g`, CERT.13 constructs

\[
  K_\perp
  =K-Kg(g^T Kg)^{-1}g^T K.
\]

This is the conditional covariance of the Gaussian discrepancy under
`g^T delta=0`. It is positive semidefinite by the Gaussian Schur-complement
identity and satisfies

\[
  K_\perp g=0.
\]

It is also the factorisation-free form of the intended resident construction.
If `K=W W^T` and `N` spans the null space of `g^T W`, then

\[
  WNN^TW^T=K-Kg(g^TKg)^{-1}g^TK.
\]

The new path therefore removes eigenvector signs, truncated eigenspaces, SVD
rank thresholds and final response-free projection roundoff without adding a
regularizer or changing the structure-wise orthogonality target.

The RBF and projected covariance are built in 512-bit Arb. The nonzero Gram
scalar must have a strictly positive lower bound, and every outward
`K_perp g` coordinate must contain zero. Failure of either condition stops the
constructor.

Orthogonal Gaussian processes address confounding between a mean structure and
a Gaussian stochastic component; the source formulation is described by
Plumlee and Joseph, *Orthogonal Gaussian Process Models*, Statistica Sinica 28
(2018):

- https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n23/j28n23.html

## 5. Validated function-space NIG posterior

Let `lambda_beta` and `lambda_delta` be the coefficient and discrepancy prior
precisions. The latent function covariance on the registered domain is

\[
  P=\lambda_\beta^{-1}gg^T
    +\lambda_\delta^{-1}K_\perp,
\]

and the prior location is `m=m_beta g`.

For the `H0` observation selector `S`, response vector `y`, and residual
`r=y-m_S`, define

\[
  C=I+P_{SS}, \qquad \alpha=C^{-1}r.
\]

For every action `i` in `A0`, CERT.13 computes

\[
  \mu_i=m_i+P_{iS}\alpha,
\]

\[
  v_i=P_{ii}-P_{iS}C^{-1}P_{Si},
\]

\[
  a_H=a_0+\frac{|H0|}{2},
  \qquad
  b_H=b_0+\frac12 r^T\alpha,
\]

and

\[
  s_i^2=\frac{b_H}{a_H}(1+v_i),
  \qquad \nu=2a_H.
\]

These are the standard function-space Gaussian/NIG identities equivalent to
the integrated linear model. Gaussian-process conditioning is documented in
Rasmussen and Williams, *Gaussian Processes for Machine Learning*, Chapter 2:

- https://gaussianprocess.org/gpml/chapters/RW2.pdf

Both solves use exactly

```text
arb_mat.solve(..., algorithm="precond")
```

at one preregistered 512-bit precision. The determinant of `C` must have a
strictly positive lower endpoint. `algorithm="approx"`, `inv`, NumPy linear
algebra, Cholesky jitter, ridge regularization, diagonal loading, result-
dependent precision escalation and fallback solves are forbidden.

The output for every `A0 x threshold` coordinate is a
`CertifiedStudentTPredictiveParameterBall` with exact outward binary endpoints
for threshold, location, scale squared and degrees of freedom. Scale squared
and degrees of freedom must remain strictly positive.

python-flint documents the validated Arb matrix solve and distinguishes its
unbounded-error `algorithm="approx"` variant:

- https://python-flint.readthedocs.io/en/latest/arb_mat.html

Arb's ball arithmetic and error-containment model are described in Johansson,
*Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic*:

- https://arxiv.org/abs/1611.02831

## 6. CERT.12 incomplete-beta zero-crossing repair

CERT.13 composition exposed a rigorous special-function edge case not covered
by the three Cauchy point identities. If a location ball is extremely narrow
around a threshold, the standardized endpoint is close to zero and the Arb
ball for

\[
  z=\frac{\nu}{\nu+x^2}
\]

may extend microscopically beyond one. Passing that enclosure directly to the
regularized incomplete beta function can return an indeterminate ball.

The repair first intersects the argument with its proved mathematical domain
`[0,1]`. It then evaluates the smaller of the complementary arguments using

\[
  I_z(a,b)=1-I_{1-z}(b,a).
\]

For `x^2 <= nu`, it evaluates

\[
  w=\frac{x^2}{\nu+x^2}
\]

and returns `1/2 +/- I_w(1/2,nu/2)/2`. Otherwise it retains the original
`z` formula. This is an exact analytic identity and domain intersection, not a
tolerance, padding, clipping of a point estimate or precision retry.

## 7. Nonnegative MAP-regret upper bound

CERT.12 used `1-2L` after proving the candidate mass lower bound `p_c>=L`.
When `L>1/2`, that expression is negative even though regret is nonnegative.
CERT.13 records the correct bound

\[
  \max\{0,1-2L\}.
\]

At `L>1/2`, the candidate is necessarily a MAP class and its regret upper bound
is exactly zero. This repair changes no threshold, failure budget, island count
or observed decision.

## 8. Sparse candidate/confirmation projector

The selection/confirmation theorem needs only the class selected by the
independent selection transcript. CERT.13 queries that one full class ID and
never constructs a vector over `6^d` classes.

For one state, every CDF interval gives a finite possible-bin set `Q_k`. For
candidate signature `c`, the indicator lower and upper values are

\[
  l(z,c)=1
  \quad\text{iff every }Q_k=\{c_k\},
\]

\[
  u(z,c)=1
  \quad\text{iff every }c_k\in Q_k.
\]

For exact state masses `w_z`, the sparse projector returns only

\[
  L_c=\sum_z w_z l(z,c),
  \qquad
  U_c=\sum_z w_z u(z,c).
\]

Boundary-compatible uncertain mass enters only `U_c`. The projector neither
chooses a nearest bin nor appends `other`, renormalizes, enumerates possible
Cartesian signatures, or materializes a posterior vector. The Gate proves
that this direct candidate query matches the corresponding class-mass query in
the complete CERT.11 sparse bounds.

For confirmation, using the median of island lower bounds is conservative:
the unobserved exact candidate-indicator median is no smaller. If the lower
median reaches the frozen `1/2` threshold, the CERT.12 fixed-candidate regret
certificate remains valid; otherwise the decision abstains.

## 9. Authorization boundary

The standalone exact-H0 algebraic constructor is the only newly authorized
object:

```text
P3F4_CERT13_STANDALONE_H0_PARAMETER_BALL_CONSTRUCTION_AUTHORIZED = true
```

All operational switches remain false:

```text
P3F4_CERT13_OPERATIONAL_H0_ACCESS_AUTHORIZED = false
P3F4_CERT13_OPERATIONAL_CDF_RESULT_ACCESS_AUTHORIZED = false
P3F4_CERT13_SPARSE_PROJECTOR_RESULT_ACCESS_AUTHORIZED = false
P3F4_CERT13_ISLAND_EXECUTION_AUTHORIZED = false
P3F4_CERT13_RESIDENT_SMC_INTEGRATION_AUTHORIZED = false
```

Every CERT.10--CERT.12 execution, product-source and result-access switch also
remains false. The guarded operational adapter stops before inspecting a
result, particle, semantic state, provider or `H0` value.

## 10. Response-free Gate identity

CERT.13 retains all 94 CERT.3--CERT.12 checks and adds 15 checks covering:

1. the complete old/new authorization boundary;
2. exact `H0` and standardizer identity binding;
3. exact polynomial-key evaluation without raw-AST or floating dependence;
4. Schur-complement RBF projection and certified orthogonality;
5. symmetric outward RBF enclosures against a higher-precision reference;
6. the zero-polynomial vacuous constraint;
7. the inactive-component exact conjugate identity;
8. the pinned validated Arb solve and forbidden numerical shortcuts;
9. complete `A0 x threshold` parameter coverage;
10. monotone outward CDF composition including the zero crossing;
11. cross-target/provider/component failure closure;
12. sparse boundary-uncertain candidate mass propagation;
13. equality with the corresponding complete sparse class query;
14. lower-bound composition with the fixed-candidate MAP certificate; and
15. pre-access operational guards and absence of result/class enumeration.

The registered identity is `109/109`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

These checks are deterministic correctness fixtures. They are not simulation,
calibration, efficacy, discovery, real-data, held-out or confirmatory evidence.

## 11. Remaining composition gap

CERT.13 proves the mathematical full-state parameter provider, but the current
resident collapsed likelihood still numerically uses the historical floating
factor basis. Joining a CERT.13 CDF to a resident particle before reconciling
those two target implementations would cross targets.

The next admissible Gate must replace the resident floating factor-basis
collapsed likelihood with the same factorisation-free function-space
covariance and prove one target identity through:

1. collapsed prior and likelihood mass;
2. Feynman--Kac bridge weights;
3. local/RJ endpoint mass and acceptance;
4. finite-`N` theorem controls; and
5. the guarded sparse fixed-candidate result adapter.

Until that source-composition Gate passes, operational `H0` access, entropy,
product streams, island execution, resident SMC, predictive calibration, real
data, acquisition, held-out access, confirmation, efficacy and paper
superiority claims remain blocked.
