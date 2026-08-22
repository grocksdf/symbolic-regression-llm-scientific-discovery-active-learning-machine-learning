# P3F.4-CERT.18 actual Arb refinement and linear normalization contract

Status: **RESPONSE-FREE SOURCE COMPOSITION; ALL OPERATIONAL EXECUTION BLOCKED**

Baseline: `1c05dfb81484f6a7eeee79f1290964e4bb964884`

CERT.18 binds the actual CERT.13/14 exact-input function-space evaluator and
the CERT.15 probability layer to CERT.17's preregistered precision sequence. It
also removes the quadratic particle-pair loop from log-mass normalization. It
does not read operational responses, particles, threshold bits, entropy, real
data, held-out data or scientific results.

## 1. Actual evaluator refinement

The following actual source path now accepts only the registered precision
rounds `p_r = 512 * 2^r`:

1. exact polynomial and component state;
2. exact frozen binary domain and `H0` identity;
3. projected RBF function-space covariance;
4. weighted Gaussian/NIG collapsed target;
5. exact local/RJ auxiliary and prior ratio;
6. outward MH acceptance probability; and
7. CERT.17 pre-bit decision envelope.

The target plan hash, state identity, bridge coordinate and proposal identity
are unchanged across rounds. Only the outward numerical enclosure changes.
Two consecutive valid enclosures are intersected exactly before the existing
CERT.17 allocation check. No threshold is observed during this process.

The Gate exercises the complete actual MH path at 512 and 1024 bits and proves
that the enclosures overlap, tighten, compose through CERT.17 and satisfy the
frozen per-comparison allocation before threshold access. This is deterministic
numerical certification on a controlled exact fixture, not a simulated or
formal scientific experiment.

## 2. Linear-time outward normalization

For log-mass intervals `[l_i,u_i]`, choose the exact common shift

\[
  s=\max_i u_i.
\]

Compute once

\[
  L=\sum_j \exp(l_j-s), \qquad
  U=\sum_j \exp(u_j-s).
\]

For each particle, outward ratio bounds are

\[
 \underline p_i=
 \left[1+\exp(s-l_i)\{U-\exp(u_i-s)\}\right]^{-1},
\]

\[
 \overline p_i=
 \left[1+\exp(s-u_i)\{L-\exp(l_i-s)\}\right]^{-1}.
\]

These are algebraically the same monotone endpoint bounds as CERT.15's
pairwise ratios, but the two endpoint sums are shared by all particles. The
implementation uses at most `4N` exponential evaluations and no `N(N-1)` pair
materialization. Time and auxiliary memory are both `O(N)`.

The Gate verifies equal-mass containment at both refinement rounds, exact shift
invariance, overlap with independently evaluated 2048-bit point and exhaustive
small endpoint expressions, absence of nested particle loops, and the full
registered `N=212408` complexity identity. It does not time a benchmark or use
generated data.

## 3. Numerical backend premise

All mathematical inputs are exact dyadic rationals, integers or fixed
mathematical constants. For each finite state the evaluation graph is finite.
The weighted system is `I + sqrt(W) P sqrt(W)` with the exact projected
covariance `P` positive semidefinite, so the exact system is strictly positive
definite. Standardizer variance and the nonvacuous projected-RBF Gram scalar
must be certified positive; otherwise the round fails closed.

The software theorem remains conditional on the pinned FLINT/Arb backend
honouring its documented inclusion semantics and increased-precision
convergence behavior. FLINT states that Arb operations return balls containing
the exact operation on all points in the input balls and recommends
exponentially increasing precision (Ziv's strategy). Its matrix-solve
documentation states that successful validated solves contain the exact
solution, while insufficient precision returns failure. See:

- https://flintlib.org/doc/arb.html
- https://flintlib.org/doc/using.html
- https://flintlib.org/doc/arb_mat.html

CERT.18 does not claim to prove third-party software free of bugs. An accepted
round requires a successful validated solve and a valid outward enclosure.
The convergence claim is pointwise for each finite exact state, not one finite
precision or runtime bound uniform over the countably open state space.

## 4. Failure and authorization boundary

- Only powers-of-two multiples of 512 bits are accepted.
- Crossed provider, target, sampling, integration, refinement, purpose or
  coordinate identities fail closed.
- No midpoint, float target, approximate solve, inverse, jitter, regularizer,
  empirical tolerance, result-selected precision, threshold extension,
  replacement seed, retry after observing bits or partial output is admitted.
- Operational refinement, threshold access, product-bit materialization,
  island execution and resident SMC remain false.
- The external ideal-bit/product-law premise remains explicit and unimplemented.

## 5. Remaining feasibility blocker

CERT.18 removes the `O(N^2)` normalization bottleneck, but it does not make the
current CERT.9 plan experimentally feasible. The frozen plan still contains
`N=212408`, `K=27`, at most 320 bridge steps and 200 rejuvenation moves, for
`368876229120` comparison coordinates. No real experiment should be launched
under that compute identity.

The next theorem phase must replace the loose finite-particle/island bound with
a sharper valid decision guarantee or a different exact estimator whose total
cost is defensible. This must be derived before seeing performance results; it
may not reduce counts post hoc, select successful islands, weaken `alpha`, or
change the scientific estimand. The product-randomness implementation premise
must also remain explicit.
