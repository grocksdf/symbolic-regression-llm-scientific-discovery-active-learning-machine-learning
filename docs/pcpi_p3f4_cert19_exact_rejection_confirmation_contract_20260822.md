# P3F.4-CERT.19 exact rejection confirmation contract

Status: **FROZEN RESPONSE-FREE CANDIDATE; USER GATE PENDING; ALL OPERATIONAL EXECUTION BLOCKED**

CERT.18 passed the user's identity-bound response-free Gate at source
`f2a124baae3915c030d3844549b9231415bb4965`.  Its actual Arb refinement and
linear-time normalization are retained, but its frozen complexity identity

\[
KNS(1+M),\qquad
(K,N,S,M)=(27,212408,320,200),
\]

contains `368876229120` comparison coordinates and is not an admissible
experimental plan.  CERT.19 replaces the decision-confirmation estimator; it
does not hide, relax, or relabel that negative feasibility result.

This Gate uses source inspection, exact rational algebra, finite enumeration
and deterministic correctness fixtures only.  It does not access a response,
particle, result, real dataset, held-out object or experimental random stream.

## 1. Root cause: three valid contracts were never composed

CERT.9 used four fixed class coordinates, error `r/4`, a union bound and 27
median islands.  CERT.12 later proved that one selection island may name an
arbitrary candidate and fresh confirmation needs only that now-fixed class
indicator with error `r/2`; its registered `alpha=1/20` budget needs nine
median islands.  CERT.16 nevertheless inherited the older CERT.9
`K=27,N=212408` compute identity.

This is conservative, not invalid, but it is no longer the sharpest registered
decision theorem.  CERT.19 first closes that composition gap and then audits
whether it is enough for execution.

## 2. Direct-confidence corollary of the fixed-path proof

The statement of Theorem 1 in Marion, Mathews and Schmidler fixes success at
`3/4`.  Its appendix proof retains arbitrary induction parameters
`delta, delta_prime` and gives

\[
P(C_s)\ge (1-\delta)^s-\frac{\delta'}{\delta}.
\]

For a path of at most `S` steps and desired failure probability
`alpha <= 1/4`, CERT.19 sets

\[
\delta=\frac{\alpha}{2S},\qquad
\delta'=\frac{\alpha^2}{4S}.
\]

Bernoulli's inequality then gives, at every step,

\[
P(C_s)\ge 1-s\delta-\frac{\delta'}{\delta}
          \ge 1-\alpha\ge\frac34,
\]

so the proof's 2-warm induction premise is preserved.  For one fixed
`|f| <= 1`, the sufficient particle count is

\[
N\ge\max\left\{
\frac{18}{E}\log\frac1{\delta'},
\frac1{2\varepsilon^2}\log\frac2{\delta'}
\right\},
\]

and each bridge requires TV mixing target `delta/N`.  This is recorded as a
derived appendix corollary, not misquoted as the published theorem statement.
The primary source is [Finite Sample Bounds for Sequential Monte Carlo and
Adaptive Path Selection Using the L2 Norm](https://arxiv.org/abs/1807.01346),
revised 25 August 2025.

For the CERT.12 decision values

\[
S=320,\quad E=4/5,\quad r=1/50,\quad
\varepsilon=r/2=1/100,\quad\alpha=1/20,
\]

the result is one confirmation island with `N=69197`, rather than nine islands
with `N=53102` each.  This removes median amplification and every class-count
union.  However, retaining the 200-step ceiling still gives
`4428608000` target evaluations.  Therefore direct confidence is a valid
baseline repair but remains **operationally infeasible**.

## 3. Alternatives rejected by the cost audit

### 3.1 Exact prior rejection

Prior rejection accepts with probability `Z/M`.  The frozen AC correctness
ledger lower-bounds this by only `2.6021757517e-5`; a fixed 26000-sample
confirmation would require about one billion proposals in expectation.  This
is mathematically exact but not a sufficient system repair.

### 3.2 Global-minorization SMC after the direct corollary

The direct corollary reduces islands and particles but not the final-open-target
minorization depth.  Charging a maximum of 200 steps is not evidence that an
actual response path satisfies that ceiling.  CERT.19 therefore continues to
block resident SMC rather than interpreting the smaller `N` as authorization.

### 3.3 Local-mixing SMC theorem

Mathews and Schmidler's [finite-sample local-mixing SMC
theorem](https://arxiv.org/abs/2208.06672) was checked from its original TeX.
Its main bound depends on a finite partition count `p`, the minimum partition
mass `mu_star` over the complete path, bounded density ratios and every
restricted-kernel mixing time.  Partitioning by the implicit operational
classes recreates the intractable class count and tiny minimum mass;
coarse component or core/tail partitions do not establish fast mixing on the
remaining countably-open AST space.  The theorem is therefore not silently
transferred to this target.

## 4. Full-support envelope rejection proposal

Let the unnormalized posterior target on raw state `z=(T,d)` be

\[
h(z)=p_0(z)L(z).
\]

CERT.3 supplies exact prior mass for every semantic-core/raw-AST lift, CERT.14
supplies outward collapsed-target balls, and CERT.2 supplies a likelihood
envelope `M` on the complete tail.  For each finite semantic-core/component
atom `i`, CERT.19 accepts outward total-mass bounds

\[
0\le \underline h_i\le h_i\le\overline h_i.
\]

The single analytic-tail atom has upper mass
`rho^J M` and lower mass zero.  An exact dyadic ticket grid gives every atom at
least one ticket and apportions all remaining tickets in proportion to the
exact rational upper bounds.  Hence proposal probabilities `q_i` are positive,
sum exactly to one and do not depend on a rounded normalization claim.

Define

\[
B=\max_i\frac{\overline h_i}{q_i},\qquad
\underline Z=\sum_i\underline h_i.
\]

Every state in atom `i` uses the already proved exact conditional raw-AST or
tail-prior lift.  Its target/proposal ratio is at most `B`.  Standard rejection
therefore accepts a proposed state with probability

\[
\frac{h(z)}{Bq(z)}.
\]

The joint mass of proposing and accepting `z` is exactly `h(z)/B`; conditional
on acceptance the output law is exactly `h/Z`.  Independent proposals give an
i.i.d. accepted posterior sequence.  The Gate verifies this identity by exact
finite rational enumeration and rejects any target mass above its outward
atom bound.

On the historical AC cost ledger, a 32-bit two-atom audit gives an acceptance
lower bound more than 15 times the prior-rejection lower bound.  This number is
a response-free historical correctness-ledger cost audit, not real-data
evidence and not a production acceptance claim.

## 5. Finite proposal cap and no favourable retry

Unbounded waiting is not an experimental budget.  For desired `n` accepted
draws, acceptance lower bound `a` and cap-failure budget `beta`, choose the
smallest integer `ell` with `2^-ell <= beta` and

\[
\mu_0=n+\ell+
\left\lceil\sqrt{\ell^2+2n\ell}\right\rceil,
\qquad
T=\left\lceil\frac{\mu_0}{a}\right\rceil.
\]

For `X ~ Binomial(T,p)` and every `p>=a`, the multiplicative Chernoff lower
tail gives

\[
P(X<n)\le e^{-\ell}<2^{-\ell}\le\beta.
\]

If fewer than `n` acceptances occur by proposal `T`, the complete confirmation
abstains.  It may not extend the cap, replace a random stream, retry the same
candidate or return a partial decision.  Because the cap event depends only on
accept/reject indicators, conditioning on completing by the cap does not alter
the i.i.d. target law of the accepted states.

For the historical AC audit only, 512 acceptances with `beta=0.01` require a
conservative cap between one and two million proposals.  This is far below the
direct-SMC worst-case count while preserving an explicit failure policy.

## 6. Exact sequential MAP confirmation

Candidate selection uses an independent transcript and is arbitrary.  Let

\[
p_0=\frac{1-r}{2}.
\]

At a finite preregistered set of accepted-sample stages `n_j`, allocate
`alpha_j=alpha/J`.  If `x_j` of the first `n_j` exact posterior draws belong to
the fixed candidate, certify only when

\[
P_{X\sim\operatorname{Bin}(n_j,p_0)}(X\ge x_j)\le\alpha_j.
\]

The tail is evaluated exactly as a rational number.  Monotonicity in the
binomial success probability and a Bonferroni bound give familywise false
confirmation probability at most `sum alpha_j = alpha`, despite stopping at
the first crossed stage.  A confirmed candidate has posterior mass greater
than `(1-r)/2`, so every competitor has mass at most its complement and the
candidate's posterior 0--1 MAP regret is at most `r`.  If no stage crosses,
the result is `abstain`; a second candidate is not tried.

The stages in the deterministic test are correctness fixtures, not a new
production freeze.  Operational stages, cap budgets and the accepted external
ideal-uniform/product-law premise must be frozen in a later Gate before any
response or experiment access.

## 7. Remaining blockers and claim boundary

CERT.19 does **not** yet authorize an operational sampler.  The following
source-composition work remains:

1. bind CERT.14 actual Arb atom-mass balls to the rational ticket builder;
2. implement exact ticket selection and conditional semantic-core/tail draws;
3. compose rejection comparisons with CERT.17/18 threshold-blind refinement;
4. prove the accepted-stream/cap state machine uses no retry or partial result;
5. freeze confirmation stages, proposal cap budgets and the selection engine;
6. explicitly accept and materialize the ideal independent-bit premise; and
7. pass a new identity-bound response-free source Gate.

Until those blockers close, target-ball access, ideal-bit materialization,
rejection execution, resident SMC, real data, acquisition, held-out access,
confirmatory materialization and paper efficacy claims all remain false.
