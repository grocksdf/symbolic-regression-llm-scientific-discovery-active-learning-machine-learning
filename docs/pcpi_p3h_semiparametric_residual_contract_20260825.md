# PCPI P3H semiparametric residual-law contract

## Why P3G is closed

P3G.1--P3G.5 changed finite mean, variance, function-energy and observed-regime
states while preserving the same finite Student-t response family. P3G.5 still
rejected 8 of 24 real prequential PIT audits. The observed year nuisance
improved aggregate Gas CO evidence but exchanged resolved seed failures for
new ones. That pattern is evidence against another finite-state patch, not a
license to tune one more state grid.

P3H therefore changes the residual distribution class itself. It does not
change datasets, seeds, response budgets, validation order, PIT bets, false
alarm allocation or the held-out boundary.

## Predictive estimand

Let \(\mathcal H_{t-1}\) contain only observations released before round
\(t\). The registered scientific engine issues a continuous base forecast

\[
 F_t(y)=P_0(Y_t\le y\mid x_t,\mathcal H_{t-1}),\qquad
 f_t(y)=F_t'(y).
\]

Its raw probability residual is

\[
 U_t=F_t(Y_t).
\]

P3H models the common innovation law of the raw PIT sequence by a distribution
\(G\) on \([0,1]\). The issued semiparametric forecast is

\[
 H_t(y)=G_t(F_t(y)),\qquad
 h_t(y)=g_t(F_t(y))f_t(y),
\]

where \(G_t\) is constructed only from \(U_1,\ldots,U_{t-1}\). This is a
semiparametric predictive composition. The finite symbolic/discrepancy model
remains the scientific posterior; \(G_t\) is not called a posterior over
scientific laws and cannot change structural class probabilities.

## Response-independent dyadic Pólya-tree sieve

At depth \(d\), the unit interval is divided into the fixed cells

\[
 B_{d,k}=[k2^{-d},(k+1)2^{-d}),\quad k=0,\ldots,2^d-1,
\]

with the last cell closed at one. Every binary split has the symmetric
Krichevsky--Trofimov prior

\[
 \theta_s\sim\operatorname{Beta}(1/2,1/2).
\]

If \(N_{s0}\) and \(N_{s1}\) are past raw-PIT counts below node \(s\), its
posterior predictive left probability is

\[
 \widehat\theta_s=
 \frac{N_{s0}+1/2}{N_{s0}+N_{s1}+1}.
\]

Multiplying these split probabilities along a leaf path gives a strictly
positive normalized leaf mass. The density is uniform inside each active
leaf, so \(G_t\) is continuous and absolutely continuous even though its
density may jump at dyadic boundaries.

The active depth is the universal count-only schedule

\[
 d_n=\left\lfloor\frac{1}{2}\log_2(\max\{n,1\})\right\rfloor.
\]

Thus \(2^{d_n}\le\sqrt n\) for \(n\ge1\), every fixed dyadic resolution is
eventually reached, and there is no fixed maximum depth, selected bandwidth,
validation-dependent stopping rule or dataset-specific residual family.
Finite-time forecasts always use a finite exact split-count calculation. The
construction follows the conjugate Pólya-tree predictive framework of
[Lavine (1992)](https://doi.org/10.1214/aos/1176348767), the tree-of-urns
foundation of [Mauldin, Sudderth and Williams
(1992)](https://doi.org/10.1214/aos/1176348766), and the binary
Beta-\(1/2\) universal predictor of [Krichevsky and Trofimov
(1981)](https://doi.org/10.1109/TIT.1981.1056331).

## Proven implementation properties

For every finite history accepted by the implementation:

1. every leaf mass is positive and the masses sum to one;
2. the CDF has endpoints zero and one, is continuous and monotone, and its
   implemented inverse composes back to the requested probability up to the
   frozen floating tolerance;
3. the transformed density obeys the exact chain rule
   \(\log h_t=\log f_t+\log g_t\circ F_t\);
4. tree posterior prediction depends on past raw-PIT counts, not their storage
   order;
5. active depth depends only on the number of past residuals;
6. the response at round \(t\) is passed to the residual state only after
   \(F_t\) has been formed from \(\mathcal H_{t-1}\);
7. the base scientific posterior update is bit-for-bit the existing update;
8. the production module has no RNG, retry, dataset identity, target identity,
   seed, rejection threshold or held-out surface.

These are algebraic and source-composition statements. The prequential
ordering follows [Dawid's forecasting formulation
(1984)](https://doi.org/10.2307/2981683): a forecast is issued from past
information, then the realized observation is used to form the next forecast.

## Leakage boundary

The current response cannot enter its own base forecast or residual CDF. A
P3H state contains the base state and only raw PIT values from completed
updates. Prediction is a pure function of that immutable state and registered
row indices. Update first reconstructs the one-row base law from the old
state, evaluates the observed raw PIT, appends it to the residual history, and
only then advances the base state.

Initial-development responses may seed both states through the same
response-independent row order. Validation responses are subsequently opened
one at a time. Candidate responses, held-out arrays and future validation
responses are absent from P3H.1 and must remain sealed in any later P3H real
calibration Gate.

## What the correctness Gate does not prove

P3H.1 uses a deterministic inference-correctness fixture, not simulated or
real efficacy data. A pass does **not** prove that the raw PIT innovations are
stationary or conditionally exchangeable, that the chosen common residual law
is adequate for UCI tasks, or that future PITs are uniform. It does not prove
acquisition correctness, superiority, held-out performance, discovery or
paper acceptance.

In particular, Pólya-tree conjugacy applies to the declared innovation model;
it does not turn empirical calibration into a theorem about an arbitrary
changing base learner. A separately frozen real prequential Gate is required.
Even a 24/24 non-rejection would mean only non-rejection against the registered
PIT betting family. Acquisition must remain blocked until that result is
audited and the transformed predictive law is incorporated into the decision
utility without pretending it is still a Student-t mixture.
