# PCPI P3K.1 shared-innovation identifiability repair

## Decision boundary

P3J.15 completed its registered real-development protocol, but its effectiveness
assessment was `REAL_ADVANTAGE_NOT_DEMONSTRATED`. That result and its P3J method
identifiers remain immutable. P3K.1 is a new no-data correctness stage. It does
not reinterpret P3J.15, access held-out responses, or authorize a real run.

P3J assigned an unrestricted residual density to every scientific class. This
is too flexible for class discovery. If the data density is (h), a class with
base CDF (F_c) and density (f_c) can use

\[
g_c(u)=\frac{h(F_c^{-1}(u))}{f_c(F_c^{-1}(u))}
\]

and obtain (g_c(F_c(y))f_c(y)=h(y)). Hence every sufficiently rich class can
explain the same response law and the residual learner can erase the scientific
distinction that acquisition is supposed to resolve.

## Repaired probability object

For each likelihood power (eta), P3K retains one strict-prefix dyadic
innovation law (G_{\eta,t}), shared by all frozen classes inside that power.
Different likelihood powers still retain different states. For class (c),

\[
q_{\eta,c,t}(y)
=g_{\eta,t}(F_{\eta,c,t}(y))f_{\eta,c,t}(y).
\]

The change of variables (u=F_{\eta,c,t}(y)) proves that every conditional
density integrates to one. Therefore
(p_{\eta,t}(c)q_{\eta,c,t}(y)) is a normalized joint distribution and preserves
the current scientific-class marginal exactly.

Sharing is an identifiability restriction, not a result-derived penalty. Since
the KT predictive CDF is strictly increasing, if two repaired conditional CDFs
are equal then
(G(F_c(y))=G(F_d(y))) implies (F_c(y)=F_d(y)). One shared law therefore
cannot independently transport two distinct class forecasts to an arbitrary
common density.

## Observable strict-prefix update

The class is a scientific latent variable and its label is never revealed.
P3K consequently does not select a class, create soft labels, or update several
counterfactual class-specific residual histories. Before admitting response
(Y_t), it computes every class CDF under the frozen state and the observable
base-mixture PIT

\[
U_{\eta,t}=\sum_c p_{\eta,t}(c)F_{\eta,c,t}(Y_t).
\]

Only this one number is appended to power (eta)'s shared residual state, after
the response has been scored. This is a prequential empirical-Bayes calibration
rule; it is not represented as a posterior over scientific laws. Candidate,
validation-future, and held-out responses have no scoring input path.

## Generalized-Bayes update coherence

For (eta<1), the old implementation scored acquisition with normalized
Student-t predictive densities but advanced structure mass with an unnormalized
powered-likelihood marginal. Those are different probability objects, so the
direct class Bayes identity was required only at (eta=1).

P3K separates the two roles. Coefficient sufficient statistics still advance
under their registered likelihood power. Structure mass advances recursively
with the exact normalized one-step Student-t density used by acquisition. The
shared calibration factor is then applied by class. For every registered power,
the resulting class probabilities must equal

\[
\frac{p_{\eta,t}(c)
      f_{\eta,c,t}(Y_t)
      g_{\eta,t}(F_{\eta,c,t}(Y_t))}
     {\sum_d p_{\eta,t}(d)
      f_{\eta,d,t}(Y_t)
      g_{\eta,t}(F_{\eta,d,t}(Y_t))}.
\]

The implementation checks this identity after every reveal. At (eta=1), the
recursive normalized update also reduces to the ordinary conjugate Bayesian
structure update.

## P3K.1 correctness claims

The hand-authored algebra fixture checks conditional normalization, preservation
of class mass, recovery of base EIG under the uniform law, a separated ranking
change under a nonidentity shared law, one-update strict-prefix ordering, the
observable mixture-PIT identity, separate states across all four powers, and the
direct class Bayes identity for every power. It also checks that the production
module has no real-data, held-out, RNG, or candidate-response surface.

These checks establish implementation and finite-model coherence only. They do
not establish residual stationarity, calibration adequacy, acquisition
efficacy, held-out performance, universal superiority, scientific discovery,
or paper acceptance. A later protocol must receive a new stage identity,
configuration hash, source tree, output path, and user-only execution command.
