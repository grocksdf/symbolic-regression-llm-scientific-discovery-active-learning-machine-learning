# P3F.2 open-target mathematical and implementation contract

Decision: **P3F.2a--c IS ADMISSIBLE AS CORRECTNESS-ONLY WORK; REAL AND
ACQUISITION EXECUTION REMAIN BLOCKED.**

## Root-cause target

P3E.3 rejected the frozen nominal predictive law for three of eight CCPP
validation-role seeds and left only five seeds eligible for a proper nominal
marginal interpretation. The response is not a seed, threshold, budget,
regularization, or variance-inflation change. P3F instead changes the model
class before any new real run:

\[
  S\sim p_{\mathcal G}(S),\qquad
  y=f_S(x;\theta_S)+d\,\delta_S(x)+\epsilon.
\]

The grammar, priors, likelihood, equivalence map, and proposal-correction rule
are frozen without accepting response values, dataset identifiers, benchmark
answers, or held-out state.

## P3F.2a: proper countably-open typed target

The registered first target uses finite, dimensionless-real ASTs with terminals
`one` and `variable`, unary `neg`, and binary `add` and `mul`. It is deliberately
small enough for exact algebraic proof; it is not claimed to cover a broad
scientific operator language.

For node count \(L\ge1\),

\[
  p(L=\ell)=(1-\rho)\rho^{\ell-1},\qquad 0<\rho<1.
\]

Let \(N_\ell\) be the exactly counted number of well-typed raw ASTs of size
\(\ell\). Conditional on size, the target is uniform:

\[
  p_{\mathcal G}(S)=\frac{(1-\rho)\rho^{|S|-1}}{N_{|S|}}.
\]

Thus the target has countably infinite support and unit mass. Exact enumeration
through \(L\le L_{\max}\) is explicitly conditional on that event. Its retained
and omitted masses are

\[
  P(L\le L_{\max})=1-\rho^{L_{\max}},\qquad
  P(L>L_{\max})=\rho^{L_{\max}}.
\]

The reference code never relabels this finite slice as the complete open
posterior.

Raw AST identity and scientific equivalence are separate. For this registered
algebraic language, every AST is mapped exactly to its multivariate integer
polynomial. Posterior class mass is the prior-mass-aware sum

\[
  p(C\mid D)=\sum_{S:\,C(S)=C}p(S\mid D).
\]

No simplifier is allowed to discard or duplicate probability mass.

The complete registered latent state is

\[
  z=(S,C(S),\theta_S,d,\lambda,\sigma^2).
\]

P3F.2 supports one explicit `gaussian-homoscedastic-nig` noise state and the
explicit `none` measurement-error state. Student-t, heteroscedastic, and latent
input-error states are rejected rather than silently approximated. Adding them
requires a new frozen target version and appropriate metadata.

## P3F.2b: exact finite-slice posterior

For the exact reference, \(f_S(x;\theta_S)=\theta_S g_S(x)\). Its design column
is also the exact parameter Jacobian, so the P3F.1 structure-wise discrepancy
constraint is tangent-correct for this one-amplitude model:

\[
  J_S^\top\delta_S=0,\qquad J_S=g_S(X).
\]

For each response-independent RBF kernel state \(K_\lambda=W_\lambda
W_\lambda^\top\), the implementation constructs

\[
  N_S=\operatorname{null}(J_S^\top W_\lambda),\qquad
  A_{S,\lambda}=W_\lambda N_S,\qquad
  K^\perp_{S,\lambda}=A_{S,\lambda}A_{S,\lambda}^\top.
\]

PSD therefore holds by construction and \(J_S^\top A_{S,\lambda}=0\) on the
registered finite domain. A single spike component is assigned mass
\(p(S)(1-\pi_d)\); slab/kernel components receive
\(p(S)\pi_d p(\lambda)\). The spike is not duplicated across kernel states.

Coefficients, discrepancy coordinates, and noise variance are integrated with
one Normal--Inverse-Gamma implementation. The resulting Student-t components
form one normalized posterior-predictive mixture. Raw-AST and equivalence-class
posterior masses, batch/sequential sufficient statistics, predictive density,
and row-order equivariance are machine checked.

## P3F.2c: collapsed sequential SMC/RJMCMC reference

The correctness engine keeps one exhaustive state for every registered
\((S,d,\lambda)\) component. This removes Monte Carlo error and isolates the
sequential update algebra. At observation \(t\), it reweights by the exact
component marginal-likelihood increment and checks

\[
  \log Z_T=\sum_{t=1}^T
  \left(\log Z_t-\log Z_{t-1}\right).
\]

Two deliberately different, fully evaluable proposals are registered:

1. complete-uniform proposal over all other collapsed states;
2. prior-independence proposal.

For proposal \(q\), the move uses

\[
  \alpha(i,j)=\min\left\{1,
  \frac{\pi(j)q(j,i)}{\pi(i)q(i,j)}\right\}.
\]

Continuous coordinates are integrated, so dimension matching has an empty
auxiliary variable and unit Jacobian. Birth, death, replacement,
discrepancy-spike, and kernel-state changes are all represented. The exact
transition matrix must satisfy detailed balance and stationarity. Changing the
proposal may affect a future sampler's efficiency, but it must not change the
frozen target.

## Evidence boundary and unresolved work

P3F.2 is not a scalable open-grammar particle engine. Its exhaustive finite
slice proves target/update identities only. It does not establish search
coverage, predictive calibration, real-data efficacy, acquisition advantage,
or scientific-law discovery.

Additional limitations are explicit:

- only a dimensionless polynomial grammar is registered;
- general nonlinear constants require a parameter-dependent tangent projection
  with its normalization and rank behavior included in the target;
- exact polynomial equivalence does not solve general transcendental identity;
- orthogonality holds on the registered finite covariate domain and makes no
  extrapolation guarantee;
- noise-family and measurement-error model averaging remain future target
  versions; and
- learned, LLM, evolutionary, or GFlowNet proposals are not admitted until
  their post-filtering proposal probability is auditable.

No new real-data command is authorized. The next admissible phase is a scalable
particle approximation whose output is checked against this exact reference,
followed by a separately frozen predictive-calibration protocol.
