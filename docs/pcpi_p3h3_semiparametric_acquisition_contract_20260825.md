# PCPI P3H.3 transformed-law acquisition contract

## Decision

P3H.3 replaces the Student-t class-EIG probability object. It does not sample
from the P3H marginal and then reuse the old Student-t information integrand.
The production method is
`p3h-marginal-preserving-kl-projection-gauss-legendre-class-eig-v1`.

This is a correctness phase. Its fixture is deterministic algebra, not a
simulated experiment or real-data efficacy evaluation. Candidate responses,
validation responses, held-out arrays and real datasets are absent. Passing
P3H.3 does not authorize an operational acquisition run.

## Why direct reuse is not EIG about the frozen posterior

For one candidate action, let structures have frozen probabilities
\(\pi_s>0\), Student-t predictive densities \(f_s\), mixture

\[
f(y)=\sum_s \pi_s f_s(y), \qquad F(y)=\int_{-\infty}^y f(z)\,dz,
\]

and operational class \(C=c(s)\). The base class responsibility is

\[
r_c(y)=\Pr(C=c\mid Y=y)
=\frac{\sum_{s:c(s)=c}\pi_s f_s(y)}{f(y)}.
\]

P3H changes the response marginal to

\[
h(y)=g(F(y))f(y),
\]

where \(g\) is the prequential dyadic residual density. The superficially
natural joint \(h(y)r_c(y)\) has class marginal

\[
\widetilde\pi_c
=\int_0^1 g(u)r_c(F^{-1}(u))\,du,
\]

which is not generally the registered \(\pi_c=\sum_{s:c(s)=c}\pi_s\).
Entropy reduction computed from that joint is therefore not information gain
about the current frozen scientific class posterior.

## Registered joint completion

The P3H marginal alone does not uniquely identify a class/response joint law.
P3H.3 makes the additional choice explicit: among joint laws with response
marginal \(h\) and class marginal \(\pi\), choose the minimum-discrimination
completion relative to \(h(y)r_c(y)\):

\[
q^*=\arg\min_{q:\ q_Y=h,\ q_C=\pi}
D_{\mathrm{KL}}\!\left(q\,\middle\|\,h r\right).
\]

For strictly positive Student-t components, positive class probabilities and
the positive KT residual density, the finite-grid kernel is strictly positive.
Strict convexity gives a unique joint matrix. It has the scaling form

\[
q^*(c\mid y)
=\frac{b_c r_c(y)}{\sum_d b_d r_d(y)}, \qquad b_c>0,
\]

with multipliers chosen so that \(\int h(y)q^*(c\mid y)dy=\pi_c\).
Alternating log-domain row and column scaling computes this I-projection. This
is the classical minimum-discrimination / I-divergence projection principle,
not a response-fitted correction; see Csiszár (1975), DOI
`10.1214/aop/1176996454`, and Sinkhorn (1964), DOI
`10.1214/aoms/1177703591`.

The modelling boundary matters: the KL projection is a conservative joint
completion selected by P3H.3. It is not logically implied by knowing the P3H
response marginal alone, and the paper must state it as such.

## Utility and deterministic computation

The acquisition utility is the mutual information of the projected joint:

\[
I_{q^*}(C;Y)
=\sum_c\int q^*(c,y)
\log\frac{q^*(c,y)}{\pi_c h(y)}\,dy.
\]

Hence \(0\le I_{q^*}(C;Y)\le H(\pi)\). With \(u=F(y)\), the P3H outcome
measure becomes \(g(u)du\). Each registered dyadic leaf is integrated by a
fixed Gauss-Legendre rule, with weights multiplied by the exact leaf mass.
Student-t mixture inverse CDF values are obtained by 64 fixed bisections inside
the minimum/maximum component-quantile bracket. There is no RNG, observed
candidate target, dataset identifier, seed, task identity, result threshold or
response-dependent stopping surface.

At every finite order the computed joint preserves both discrete marginals to
the registered projection tolerance. Under the standard additional numerical
assumptions—positive continuous component densities, positive class masses,
existence of the continuous I-projection, and uniform integrability of its
information density—the nested rules converge to the continuous projected
mutual information. P3H.3 tests finite-grid identities and convergence
diagnostics; it does not elevate those regularity assumptions into a theorem
about arbitrary posterior misspecification.

## Error and ranking boundary

The reported numerical radius is four times the fine/coarse score difference,
plus projection residual and floating-point allowance. It is an asymptotic
deterministic diagnostic, not a rigorous finite-order quadrature bound and not
a confidence interval. Adaptive ranking returns an action only when one such
interval dominates every eligible competitor. At the maximum registered order,
overlap returns `selected_action_index=None`; ties are not broken with response
bits, retries or hidden fallback utility. The production scoring route raises a
terminal numerical error on the same condition, because its legacy return type
cannot represent abstention without selecting an action downstream.

## Frozen correctness decisions

The deterministic P3H.3 Gate checks:

1. exact P3H outcome masses on the quadrature grid;
2. exact frozen class-posterior masses after projection;
3. nonnegative, class-entropy-bounded mutual information;
4. numerical reduction to independently integrated Student-t EIG when
   \(g\equiv1\);
5. a nonidentity residual law changes the utility rather than reusing the old
   score;
6. positive affine response invariance;
7. deterministic refinement and interval-dominant selection;
8. no selection for identical unresolved candidates;
9. source-level absence of RNG, response, real-data and old-EIG dispatch
   surfaces.

The hand-authored finite-mixture fixture is an inference-correctness object. It
does not estimate performance and is not paper evidence.

## What a pass does not establish

A pass does not prove residual stationarity, real-data calibration, posterior
adequacy, strict utility-order correctness at finite quadrature, acquisition
efficacy, held-out improvement, scientific discovery or paper acceptance. It
also does not authorize silently feeding one residual law into multiple
likelihood-power ambiguity models: each operational model would need its own
prequentially valid residual state or a separately justified sharing rule.

Any later real acquisition phase must freeze that model-to-residual-state
mapping, rerun P3H.1--P3H.3 before data access, preserve pool-oracle response
sealing, publish unresolved decisions as abstentions, and be executed only by
the user.
