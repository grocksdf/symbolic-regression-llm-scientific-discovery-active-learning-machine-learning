# P3K.8 failure audit and P3L information-risk freeze (2026-09-05)

## Decision

P3K.8 is retained as a complete, valid negative real-development experiment.
It completed 96/96 registered runs with verified official data hashes, a closed
held-out split, matched budgets, and an intact evidence registry.  Its efficacy
assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED`; it is not relabelled, overwritten,
or excluded.

The next development comparison is P3L.2.  It changes one statistical object:
the PCPI acquisition objective.  All datasets, seeds, splits, observation and
candidate budgets, baselines, posterior ambiguity models, initial frozen class
target, shared prequential semiparametric response law, representative projection,
runtime identity, and assessment rules remain those of P3K.8.

## Failure localized by P3K.8

The P3K.8 numerical and protocol mechanisms worked: all PCPI rankings were
certified, all decisions passed the registered source checks, and the projected
representative guard was obeyed.  Nevertheless, its primary paired class-entropy
gain against random was negative on CCPP and uncertain on the Gas family.  CCPP
also showed large response-level entropy increases despite positive predicted
mean class information.

This is not a contradiction.  Expected information gain is

\[
I(C;Y_x)=H(C)-\mathbb E[H(C\mid Y_x)],
\]

so it controls the mean under the acquisition model.  It does not control the
lower tail of the realized entropy reduction

\[
G_x(Y_x)=H(C)-H(C\mid Y_x).
\]

Under misspecification, a high model expectation can coexist with frequent or
large negative realized gains.  The P3K.8 assessment separately required a
negative-transfer rate no larger than 0.25; the production objective did not
encode that risk boundary.

## P3L objective

Let \(Q_{x,\eta}\) be the same normalized class-conditional semiparametric joint
response law already used by P3K, for likelihood power
\(\eta\in\{0.125,0.25,0.5,1\}\).  For lower-tail mass \(\alpha=0.25\), define

\[
\operatorname{LCVaR}_{\alpha,Q}(G)
=\frac{1}{\alpha}\int_0^\alpha F^{-1}_{G,Q}(u)\,du.
\]

P3L ranks each visible candidate by

\[
U_{\mathrm{P3L}}(x)
=\min_{\eta}\operatorname{LCVaR}_{0.25,Q_{x,\eta}}(G_x).
\]

The selected candidate maximizes this finite-model lower envelope inside the
unchanged covariate-only P3K representative set.  Thus P3L optimizes the worst
lower tail directly.  Mean class information uses the nonnegative pointwise-KL
integrand on the same response quadrature and is exported only as an audit
quantity.  At finite quadrature order that integral and the discrete mean of
the entropy-reduction atoms can differ; they converge to the same mutual
information but are not asserted to be numerically identical at a finite look.

For a weighted deterministic response grid, the implementation sorts entropy
reductions and integrates exactly the lowest alpha mass, splitting the boundary
atom when necessary.  Fine and coarse nested grids provide the same numerical
interval-dominance protocol used by the predecessor.  This is a deterministic
numerical ranking certificate; it is not a frequentist confidence sequence or
a proof that the statistical model is correctly specified.

If lower-tail CVaR is nonnegative, then the model assigns at most alpha mass to
strictly negative entropy reduction: if more than alpha mass were negative, the
entire lowest-alpha quantile would be negative and its mean would be negative.
P3L maximizes this certificate quantity but does not falsely assert that a
nonnegative candidate always exists.

## Why alpha is not post-hoc tuning

The value 0.25 is copied exactly from the P3K.8 preregistered
`negative_transfer_rate_max`.  It is not estimated from P3K.8 scores, individual
datasets, seeds, responses, validation metrics, or held-out data.  The failed
run identified a mismatch between estimand and utility; it did not supply a
fitted coefficient or threshold.

## Leakage and identity boundary

- Candidate scoring receives candidate covariates, the frozen target partition,
  the strict-prefix posterior family, and its strict-prefix residual states.
- The entropy-gain distribution is integrated before any candidate response is
  opened.
- All four likelihood-power states are completed before the maximin reduction.
- Checkpoints bind the response-grid order, alpha, posterior components,
  residual-state hash, target-partition hash, and contiguous action prefix.
- No proper checkpoint prefix can release scores.
- Progress telemetry uses immutable, identity-bound event files, so a monitoring
  reader never competes with the scientific process to replace one mutable file.
- The oracle is called only after a durable decision identifies one candidate.
- Validation is evaluated only after the corresponding reveal ledger is durable.
- Held-out remains closed.

## Evidence boundary

P3L.1 uses hand-authored deterministic algebraic fixtures solely for correctness:
weighted CVaR atoms, complete-grid checkpoint recovery, maximin composition, and
score-before-reveal source isolation.  It is not simulated efficacy evidence.

P3L.2, if executed by the user, is one real-development experiment.  A negative
or inconclusive result remains a NO-GO.  Even a positive result would support only
the registered measured-pool claim and would not establish open-grammar discovery,
physical intervention, or held-out confirmation.

## Literature relation

The change follows the robust Bayesian experimental-design principle that an
acquisition objective should remain conservative over an explicit ambiguity set,
while addressing the distinct outcome-tail risk left uncontrolled by mean EIG.
It is also consistent with generalized-Bayesian design work showing that standard
EIG can be brittle under outcome-model misspecification.  P3L does not import a
paper-specific neural estimator or tune an ambiguity radius from P3K.8; it uses
the repository's existing finite ambiguity family and exact registered risk tail.
