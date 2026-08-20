# P3F.4-CERT.12 rigorous CDF kernel and split-island MAP contract

Status: **RESPONSE-FREE GATE PASSES; OPERATIONAL ORACLE AND ALL EXECUTION REMAIN BLOCKED**

GitHub baseline: `2b72f1abe36e0fd180dfb41599b02c6ee1c878a9`

CERT.12 resolves two source/theorem defects exposed by CERT.11 without running
an island or resident SMC. It implements and audits a rigorous Arb Student-t
CDF special-function kernel, and it replaces CERT.9's computationally
intractable all-class union budget for the MAP decision with independent
candidate selection and fixed-candidate confirmation.

This Gate performs only source inspection, exact rational finite
combinatorics, complete small-simplex implications, and preregistered analytic
rigorous-numerics fixtures. It does not access responses, particles, resident
results, real data, held-out data, acquisition, or a formal/confirmatory
experiment.

## 1. The full-class CERT.9 budget is not the right MAP theorem

For `d=7*|A0|` CDF coordinates, CERT.11 defines the complete implicit class
space

\[
  \mathcal C=\{0,\ldots,5\}^{d},\qquad |\mathcal C|=6^d.
\]

CERT.9 assigned each class coordinate error `r/|C|`, then union-bounded the
independent-island median failure over all classes. Because the registered
particle lower bound contains `1/(2 epsilon^2)`, substituting the complete
CERT.11 count makes the particle requirement scale quadratically in `|C|`.
The island count also grows through the all-coordinate union bound. This is a
valid but unusably conservative sufficient condition.

It cannot simply be replaced by the direct CERT.10 MAP identity
`regret <= 2*epsilon` while using the same islands to choose the empirical
argmax. The Marion--Mathews--Schmidler theorem is stated for a fixed bounded
functional `f`; an empirical argmax is a random functional selected from the
same data. A pointwise theorem does not become a uniform theorem over a random
class merely because every individual class indicator is bounded.

The registered theorem source is:

- Marion, Mathews and Schmidler, *Finite sample complexity of sequential Monte
  Carlo estimators on multimodal target distributions*, revised 2025,
  Theorem 1: https://arxiv.org/pdf/1807.01346

CERT.12 preserves that fixed-functional scope instead of silently
strengthening it.

## 2. Independent candidate selection and confirmation

CERT.12 registers disjoint product-coordinate roles:

1. one **selection** island may return an arbitrary measurable candidate class
   `c(S)` from its own transcript `S`;
2. fresh **confirmation** islands estimate only

   \[
      f_{c(S)}(z)=\mathbf 1\{C_\star(z)=c(S)\};
   \]

3. conditional on `S`, the candidate is fixed and the confirmation product
   coordinates are independent of `S`.

Therefore the fixed-functional theorem applies conditionally to every
confirmation island. If the per-island failure probability is at most `1/4`,
the exact probability that a majority of `K` independent confirmation islands
fail is

\[
  b_K=\sum_{j=(K+1)/2}^{K}{K\choose j}
      \left(\frac14\right)^j
      \left(\frac34\right)^{K-j}.
\]

For the registered failure budget `alpha=1/20`, the smallest odd count is

\[
  K=9,\qquad b_9=\frac{6413}{131072}\approx0.0489273<0.05.
\]

There is exactly one confirmed indicator, so there is no multiplication by
`|C|`. Exact finite conditional laws with arbitrary selection probabilities,
candidate maps and candidate-specific failure probabilities at most `1/4`
are checked directly.

Selection quality affects power--a poor candidate will usually cause
abstention--but it cannot invalidate a certificate because confirmation is
fresh. There is no adaptive second candidate, retry, replacement island,
favourable-key selection, or reuse of a selection coordinate.

## 3. Dimension-free MAP regret implication

Let `r` be the frozen MAP-regret budget and set the fixed-functional
confirmation error to

\[
  \varepsilon=\frac r2.
\]

If the confirmation median for candidate `c` is `m_c`, then outside the
registered failure event

\[
  p_c\ge m_c-\varepsilon.
\]

CERT.12 certifies only when `m_c >= 1/2`. Hence

\[
 p_c\ge\frac12-\frac r2=\frac{1-r}{2}.
\]

Every competing class has probability at most `1-p_c`, so

\[
 \max_j p_j-p_c
 \le (1-p_c)-p_c
 =1-2p_c
 \le r.
\]

If `m_c < 1/2`, the output is `abstain` and no MAP-regret claim is returned.
The threshold is an algebraic consequence of the preregistered `r/2` error;
it is not selected from observed performance.

The response-free Gate checks this implication on every three-class rational
simplex point with denominator 10, every candidate coordinate, and every
median grid point with denominator 20 satisfying the assumed error event.
That is a complete finite implication check, not a simulation or efficacy
experiment.

For both `|C|=6^7` and `|C|=6^700`, the registered per-island particle lower
bound and confirmation-island count are identical. The class-space size is
hashed into the plan identity but never enters the particle tolerance,
failure union or a loop.

## 4. Product-coordinate source contract

`ResidentSplitPhiloxProductSourceContract` binds the ordered selection
coordinate followed by all ordered confirmation coordinates. Every coordinate
has a distinct stable hash under one product-law identity. The future source
algorithm remains direct

```python
np.random.Generator(np.random.Philox(key=k_i, counter=0))
```

with one external 128-bit key per coordinate and no root derivation,
`SeedSequence.spawn`, jump, coordinate reuse, collision retry, replacement or
favourable key selection.

As in CERT.11, source inspection cannot prove that physical operating-system
entropy calls are mathematically independent. The external premise remains
`external-independent-os-entropy-key-tuple`; product-source materialization is
still unauthorized.

NumPy's direct Philox-key interface is documented at:

- https://numpy.org/doc/2.3/reference/random/bit_generators/philox.html

## 5. Exact Arb input contract

The numerical kernel accepts only `CertifiedDyadicInterval` endpoints. Each
endpoint is represented exactly as

\[
  m2^e,
\]

using an integer mantissa and exponent. A Python float may be recorded only
through `float.as_integer_ratio()`, which identifies the exact stored binary
number; it is never reinterpreted as a shorter decimal. Non-dyadic endpoints
are rejected by this interface.

The predictive parameter record contains outward intervals for:

- response threshold;
- predictive location;
- predictive scale squared, whose lower endpoint must be strictly positive;
  and
- degrees of freedom, whose lower endpoint must be strictly positive.

The kernel contract pins:

```text
python-flint == 0.8.0
working precision = 256 bits
precision schedule = one preregistered pass
```

There is no result-dependent precision retry. If an interval is too wide to
identify one CERT.11 bin, the existing boundary-uncertain propagation is used;
the kernel does not choose a nearest bin.

python-flint documents that `arb((m,e))` accepts exact `m*2^e` input, that Arb
operations return rigorous balls, and that `ctx.workprec` scopes the working
precision:

- https://python-flint.readthedocs.io/en/latest/arb.html
- https://python-flint.readthedocs.io/en/latest/general.html

## 6. Student-t CDF through regularized incomplete beta

For standardized value

\[
  x=\frac{t-\mu}{\sqrt{s^2}},
\]

and degrees of freedom `nu`, define

\[
  z=\frac{\nu}{\nu+x^2}.
\]

The kernel evaluates

\[
F_\nu(x)=
\begin{cases}
\tfrac12 I_z(\nu/2,1/2), & x<0,\\
\tfrac12, & x=0,\\
1-\tfrac12 I_z(\nu/2,1/2), & x>0,
\end{cases}
\]

using `arb.beta_lower(..., regularized=True)`. python-flint documents this as
the regularized lower incomplete beta function:

- https://python-flint.readthedocs.io/en/latest/arb.html#flint.arb.beta_lower

For interval `x=[x_l,x_u]`, Student-t CDF monotonicity in `x` gives the lower
bound from `F(x_l)` and the upper bound from `F(x_u)`. The same degrees-of-
freedom ball is propagated through each endpoint computation. Arb output
balls are converted to exact outward rational endpoints only through
`lower()`, `upper()` and `man_exp()`. Intersecting the resulting enclosure
with the exact mathematical CDF codomain `[0,1]` is a valid set intersection,
not normalization or point-estimate clipping.

Arb's design and rigorous ball-arithmetic guarantees are described in:

- Fredrik Johansson, *Arb: Efficient Arbitrary-Precision Midpoint-Radius
  Interval Arithmetic*, IEEE Transactions on Computers 66(8), 2017:
  https://arxiv.org/abs/1611.02831
- FLINT/Arb documentation: https://flintlib.org/doc/index_arb.html

## 7. Preregistered analytic numerical checks

The Gate evaluates only analytic Student-t/Cauchy identities with exact balls:

\[
 F_1(-1)=\frac14,\qquad F_1(0)=\frac12,\qquad F_1(1)=\frac34.
\]

Every exact value lies inside the returned Arb enclosure, and the two
nontrivial enclosure widths are below `2^-240`. It also propagates
`x in [-1,1]` and a positive scale-squared interval, confirming that the
outward result spans every analytic endpoint it must contain.

These are numerical-proof fixtures for a special-function implementation.
They are not simulated data, a model comparison, a discovery run, or evidence
of PCPI efficacy.

## 8. Full-state parameter-ball boundary remains open

`OpenTargetParticleSnapshot` currently stores NumPy floating arrays for the
design, posterior mean/covariance, noise shape and scale. Its
`predictive_cdf` uses an ordinary SciPy Student-t point evaluation. Wrapping
those rounded values in zero-radius balls would certify only the rounded
snapshot computation, not the mathematical `H0` predictive CDF.

The missing provider must instead reconstruct outward parameter balls from
the frozen mathematical inputs and validated linear algebra. The resident
design also contains a structurewise projected RBF discrepancy basis whose
exponentials and eigenspace construction require their own interval proof.
CERT.12 does not hide that dependency.

`CertifiedPredictiveParameterBallProvider` is therefore a protocol only. The
operational oracle constructor rejects any provider that treats rounded
snapshot arrays as exact, and its call guard precedes particle or provider
access. Consequently:

```text
full_state_parameter_ball_provider_authorized = false
operational_cdf_oracle_run_authorized = false
CERT.11 certified_cdf_oracle_implementation_authorized = false
CERT.11 projector_result_access_authorized = false
```

FLINT's `arb_mat.solve`/`arb_mat.inv` validated ball-linear-algebra interfaces
are candidate building blocks for the next Gate. Their explicitly approximate
algorithm variants are not admissible:

- https://python-flint.readthedocs.io/en/latest/arb_mat.html

## 9. Unchanged execution boundary

The five CERT.12 guards are false:

```text
P3F4_CERT12_FULL_STATE_PARAMETER_BALL_PROVIDER_AUTHORIZED = false
P3F4_CERT12_OPERATIONAL_CDF_ORACLE_RUN_AUTHORIZED = false
P3F4_CERT12_SPLIT_PRODUCT_SOURCE_MATERIALIZATION_AUTHORIZED = false
P3F4_CERT12_SPLIT_ISLAND_EXECUTION_AUTHORIZED = false
P3F4_CERT12_MAP_RESULT_ACCESS_AUTHORIZED = false
```

All CERT.10 and CERT.11 execution/result guards remain false. No entropy is
captured, no production Philox stream is instantiated, no particle is read,
no island is run, and resident SMC remains blocked.

## 10. Response-free Gate identity

CERT.12 retains all 79 CERT.3--CERT.11 checks and adds 15 checks covering:

1. every old and new authorization guard;
2. the exact pinned dependency in both manifests;
3. dyadic mantissa/exponent round trips and rejection of decimal surrogates;
4. immutable Arb kernel identity and forbidden point-padding flags;
5. analytic Cauchy CDF enclosures;
6. outward propagation of parameter intervals;
7. incomplete-beta, fixed-precision and exact endpoint source composition;
8. operational-oracle identity binding and pre-access guard;
9. rejection of rounded resident arrays as mathematical parameter balls;
10. dimension-free particle/island budgets for `6^7` and `6^700` classes;
11. disjoint ordered selection/confirmation product coordinates;
12. exact finite conditional failure laws without a class union bound;
13. the complete registered small-simplex MAP-regret implication;
14. frozen threshold certification versus explicit abstention; and
15. absence of class enumeration, retry, normalization or execution
    smuggling.

The registered identity is `94/94`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

## 11. Next admissible phase

The next phase is a separate response-free Gate for the full `H0` state to
predictive-parameter-ball map, including certified construction of the
structurewise RBF discrepancy design and validated posterior linear algebra.
It must then provide a sparse candidate/confirmation projector that preserves
CERT.11 boundary uncertainty and never exposes a fictitious fixed vector over
the entire implicit class space.

Only after those proofs pass may operational CDF/projector result access be
considered. Product-source materialization, island execution and resident SMC
remain separate later authorization decisions; no experiment is authorized by
CERT.12.
