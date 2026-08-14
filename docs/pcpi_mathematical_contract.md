# PCPI mathematical contract (P0 freeze)

Status: **P3C.1 real efficacy failed; P3D.1 correctness passed; P3D.2 real-only
reference-dominance protocol implemented and awaiting local execution**
Contract version: `pcpi-p3d2-reference-dominance-real-v1`
Primary target: predictive-equivalence-class identification  
Secondary target: posterior predictive risk

This document defines PCPI independently of the legacy discovery runtime. Code that
does not implement this contract must not describe scores, populations, or candidate
rankings as a PCPI posterior.

## 1. State, actions, observations, and history

At sequential step \(t\), the inferential history is

\[
H_t = H_0 \cup \{(a_i,x_i,y_i)\}_{i=1}^{t}.
\]

`H_0` contains only the labelled development observations admitted by the frozen
split manifest. Validation data are for development-time assessment and configuration
selection; they are not silently multiplied into the posterior. An acquisition-pool
action reveals exactly one previously hidden measured response. Untouched-heldout data
are outside the process that constructs, updates, or inspects \(H_t\).

## 2. Joint posterior target

The ordinary Bayesian target used by P1--P3A is

\[
\pi_t(S,\theta_S,\psi)
=p(S,\theta_S,\psi\mid H_t)
\propto p(S)\,p(\theta_S\mid S,\psi)\,p(\psi)
\prod_{(a,x,y)\in H_t}p(y\mid a,x,S,\theta_S,\psi).
\]

The structure marginal is

\[
p_t(S)=\int\pi_t(S,\theta_S,\psi)\,d\theta_S\,d\psi.
\]

No validation fitness, Pareto score, evolutionary rank, LLM confidence, or heuristic
candidate frequency is a substitute for either quantity.

P3B.5--P3B.7 separately declare the power-likelihood generalized posterior

\[
\pi_{t,\eta}(S,\theta_S,\psi)
\propto p(S)p(\theta_S\mid S,\psi)p(\psi)
\prod_{(a,x,y)\in H_t}p(y\mid a,x,S,\theta_S,\psi)^\eta,
\qquad 0<\eta\leq1.
\]

It must be named generalized Bayes. It cannot be presented as the ordinary
posterior above unless \(\eta=1\).

For P3B.6/P3B.7, let \(\Phi_S^{(0)}(x)\) denote raw basis rows and let
\(T_0\) be the termwise center/scale transform fitted from initial-development
covariates only. The posterior target is parameterized by

\[
\Phi_S(x)=T_0\{\Phi_S^{(0)}(x)\}.
\]

The same \(\Phi_S\) is mandatory for fitting, marginal likelihoods, posterior
predictive distributions, operational classes, validation metrics, class-EIG,
uncertainty, QBC, and the PCPI fallback. A coefficient posterior fitted under
\(\Phi_S\) cannot be evaluated next to raw rows \(\Phi_S^{(0)}\).

## 3. Frozen probability model

### 3.1 Symbolic structure prior

`S` is a canonical AST from a finite, versioned grammar with a fixed operator
whitelist, maximum node count, maximum depth, and dimensional-validity rules. Algebraic
duplicates are merged before assigning prior mass. The normalized prior is

\[
p(S)=\frac{\widetilde p_G(S)}{\sum_{S'\in\mathcal S_G}\widetilde p_G(S')},\qquad
\log\widetilde p_G(S)=
\sum_{v\in\operatorname{nodes}(S)}\log\rho_{\tau(v)}
-\lambda_n|S|-\lambda_d\operatorname{depth}(S).
\]

The grammar probabilities \(\rho\), penalties \(\lambda_n,\lambda_d\), support limits,
and canonicalization version are configuration values frozen before an experiment.
They may not depend on dataset name, filename, target name, an oracle expression, or
the observed validation outcome.

### 3.2 Continuous parameters

For \(d(S)\) declared coefficients,

\[
\theta_S\mid S,\sigma^2\sim
\mathcal N(0,\sigma^2\Lambda_S^{-1}),
\]

where \(\Lambda_S\) is a fixed positive-definite diagonal precision determined only by
coefficient roles in the frozen grammar. Coefficient ordering is part of the canonical
structure definition.

### 3.3 Noise parameter

The P0 nuisance parameter is \(\psi=\sigma^2\), with

\[
\sigma^2\sim\operatorname{InverseGamma}(\alpha_0,\beta_0),
\]

using density proportional to
\((\sigma^2)^{-\alpha_0-1}\exp(-\beta_0/\sigma^2)\). Hyperparameters are frozen in
the model configuration.

### 3.4 Likelihood decision

The observation model remains a homoscedastic Gaussian likelihood:

\[
y\mid a,x,S,\theta_S,\sigma^2
\sim\mathcal N(f_S(a,x;\theta_S),\sigma^2).
\]

P1--P3A use the ordinary likelihood. P3B.5--P3B.7 raise this likelihood to the frozen
power \(\eta\) and therefore uses generalized Bayes. Student-\(t\) and
heteroscedastic variants remain out of scope until separately specified and
tested. Development-only transformations must be frozen and included in the
configuration hash; held-out statistics cannot define them.

P3B.5--P3B.7 consider the fixed dimension-only candidate set
\(\{0.125,0.25,0.5,1\}\). It selects one \(\eta\) per dataset and seed by
prequential posterior-expected posterior-randomized log loss on the initial
development observations,

\[
R_n(\eta)=\sum_{i=1}^{n}
E_{(S,\theta_S,\sigma^2)\sim\pi_{i-1,\eta}}
[-\log p(y_i\mid x_i,S,\theta_S,\sigma^2)].
\]

Ties choose the largest \(\eta\). Selection occurs once before
any acquisition policy runs; every policy shares the selected value and its
calibration hash. Validation responses, pool responses, dataset names, target
names, and untouched-heldout cannot enter this selection.

Before posterior fitting, every non-intercept term in the closed real bank is
centered and scaled using only the initial-development covariates. The transform
is frozen before response acquisition and shared by all structures and policies.
It changes the prior parameterization, not the candidate subspaces, observation
budget, or likelihood. Its hash is part of the posterior target identity.

## 4. Sequential update

For a newly acquired observation \(o_t=(a_t,x_t,y_t)\), the ordinary update is

\[
\pi_t(z)=
\frac{p(y_t\mid a_t,x_t,z)\pi_{t-1}(z)}
{\int p(y_t\mid a_t,x_t,z')\pi_{t-1}(z')\,dz'},
\qquad z=(S,\theta_S,\sigma^2).
\]

Batch evaluation on \(H_t\) and the ordered sequence of updates must agree within the
registered numerical tolerance.

For P3B.5--P3B.7, the incremental generalized-Bayes factor is
\(p(y_t\mid a_t,x_t,z)^\eta\). Batch and sequential powered updates must agree
with each other and with independent high-precision noise-variance quadrature.

## 5. Exact finite-bank reference on measured observations

The reference universe is finite:

\[
\mathcal S_{\mathrm{ref}}=\{S_1,\ldots,S_K\}.
\]

For the first reference engine, every \(f_S\) is linear in its coefficients,
\(f_S(a,x;\theta_S)=\phi_S(a,x)^\top\theta_S\). The Normal–Inverse-Gamma model then
permits independently checked analytic marginal likelihoods and posterior predictives.
High-precision quadrature is an independent implementation cross-check.
The P3B.5--P3B.7 real bank is a column-order-invariant hierarchy containing intercept,
linear, additive quadratic, full quadratic with all pairwise interactions,
additive cubic, and full quadratic plus additive cubic structures. Its
generation rule depends only on input dimension.
Evidence is separated by role. Registered, provenance-verified measured observations
are required for efficacy claims. Small, fully specified controlled fixtures
may enter `EvidenceRegistry` and Gate packages only under the role
`inference_correctness_diagnostic_fixture`; they may support P1/P2/P3 numerical
correctness but never scientific-discovery efficacy, real-data advantage, or a
new-law claim. Historical generated-observation efficacy packages remain
invalidated. Disposable unit-test arrays remain outside all result packages.

### 5.1 P2A.1 collapsed and tempered target sequence

For linear-in-parameter finite-bank structures, P2A.1 integrates
\((\theta_S,\sigma^2)\) analytically and propagates the structure marginal.
The incremental potential is

\[
g_t(S)=p(y_t\mid a_t,x_t,S,H_{t-1}).
\]

When a full update is too concentrated, adaptive bridges target

\[
\eta_{t,\beta}(S)\propto p_{t-1}(S)g_t(S)^\beta,
\qquad 0\leq\beta\leq1.
\]

Bridge temperatures are selected from a frozen conditional-ESS target. Each
bridge uses corrected incremental weights and an invariant collapsed MH move.
At \(\beta=1\), exact conditional parameter/noise draws reconstruct particles
from the joint posterior. Tempering changes only the numerical path, not the
terminal posterior target.

## 6. Posterior predictive

For a structure,

\[
p_t(y\mid a,x,S)=\int p(y\mid a,x,S,\theta_S,\psi)
p_t(\theta_S,\psi\mid S)\,d\theta_S\,d\psi.
\]

The model-averaged predictive is

\[
p_t(y\mid a,x)=\sum_S p_t(S)p_t(y\mid a,x,S).
\]

All reported posterior predictions and credible intervals must derive from these
distributions, not from an unweighted top-k list.

## 7. Primary predictive-equivalence definition

The primary definition is **finite-action operational equivalence**. Let
\(\mathcal A_t^{\mathrm{op}}\) be the frozen, selection-visible action/covariate pool at
step \(t\). For structure \(S\), let \(Q_{t,S}(u\mid a)\) and
\(V_{t,S}(a)\) be its posterior-predictive quantile and variance. For frozen
quantile levels \(\mathcal U=\{0.1,0.5,0.9\}\), define the uncertainty-scaled
finite-action distance

\[
d_t(S,S')^2=\frac{1}{|\mathcal A_t^{\mathrm{op}}||\mathcal U|}
\sum_{a\in\mathcal A_t^{\mathrm{op}}}\sum_{u\in\mathcal U}
\left[
\frac{Q_{t,S}(u\mid a)-Q_{t,S'}(u\mid a)}
{\sqrt{\{V_{t,S}(a)+V_{t,S'}(a)\}/2}}
\right]^2.
\]

The operational classes are the deterministic complete-link partition whose
within-class diameter is at most a preregistered \(\epsilon_B\). Complete linkage
prevents epsilon chaining and returns disjoint, exhaustive blocks even though
raw pairwise epsilon proximity need not be transitive. Actions are sorted
canonically before hashing; cluster ties are resolved by stable structure ID.
P3B.7 resolves equivalence at the registered future measurement budget \(B\):

\[
\epsilon_B=\frac{1}{\sqrt B}.
\]

Thus the root-budget aggregate standardized separation is at most one. This
resolution is fixed by the observation budget and predictive metric, not by a
dataset, target, formula, held-out value, or observed effectiveness. Exact
predictive equivalence and alternative preregistered aggregate resolutions are
sensitivity analyses, not result-conditioned replacements for the primary
definition.

If \(C_t(S)\) is the resulting class label, class mass is

\[
P_t(c)=\sum_{S:C_t(S)=c}p_t(S).
\]

The initial posterior and visible operational domain define one frozen
structure-to-class map \(C_0:S\mapsto c\). At later rounds, the map is not
reclustered: current structure mass is pushed through the same map,

\[
P_t^{(0)}(c)=\sum_{S:C_0(S)=c}p_t(S).
\]

Both acquisition and cross-policy outcome comparison target this one random
variable. The primary endpoint is

\[
G_B=H(C_0\mid H_0)-H(C_0\mid H_B),
\]

so every utility and entropy difference refers to the same estimand. A
recomputed \(C_t\) may be recorded as a posterior diagnostic, but it cannot
define the primary sequential utility or total class-information endpoint.

## 8. Acquisition target and naming rule

The ideal class information gain is

\[
\operatorname{EIG}^{(0)}_t(a)=H(C_0\mid H_t)
-\mathbb E_{Y\sim p_t(Y\mid a)}
[H(C_0\mid H_t,a,Y)].
\]

An implementation may be called class-EIG only after comparison with exact EIG in the
reference universe, including bias, variance, rank agreement, top-k overlap, Monte
Carlo convergence, and computation cost. Until then, a predictive-variance or committee
score is named `posterior-disagreement acquisition` only if it uses posterior weights;
an unweighted candidate variance is named `candidate-disagreement acquisition`.

For real-pool argmax selection, P3B.8 retains P3B.3's adaptive Student-t-measure
Gauss--Jacobi evaluation budget from 32 to 512. Each fine rule is paired with
the half-budget rule. Four times their absolute discrepancy, plus a numerical
roundoff floor, is recorded as an asymptotic error envelope. The point-estimate
leader is certified only if its lower envelope exceeds every competitor's
upper envelope. The actual budget, coarse budget, envelope, look count, and
certificate status are evidence fields. This is an exact-reference-validated
asymptotic numerical diagnostic, not a rigorous quadrature bound or a
finite-sample confidence sequence.

Let \(X^\star\) be uniformly distributed over the registered,
selection-visible action domain and let \(Y^\star\) be its future response.
P3B.8 targets the chain-rule identity

\[
I(C_0,Y^\star;Y_a\mid H_t)
=I(C_0;Y_a\mid H_t)+I(Y^\star;Y_a\mid C_0,H_t).
\]

The first term is the exact-reference-validated class-EIG estimator. For the
second term, each class-conditional finite posterior mixture is matched in its
first two joint predictive moments and the resulting bivariate Gaussian mutual
information is averaged uniformly over the registered target actions. This
term is exact for a Gaussian class-conditional posterior predictive and is
explicitly named a Gaussian-moment surrogate for the Student-t mixtures used
in production. Both terms use nats, and the chain rule introduces no fitted
tradeoff coefficient.

P3B.8 uses the following decision rule under the shared calibrated generalized
posterior:

1. estimate class-EIG adaptively and add the deterministic conditional
   predictive-information term when certifying the joint-score leader;
2. if the joint leader is certified, select it;
3. otherwise select the action maximizing posterior epistemic variance of the
   latent mean,

\[
V_t^{\mathrm{epi}}(a)
=\operatorname{Var}_{S,\theta_S\mid H_t}[f_S(a;\theta_S)].
\]

The fallback integrates within-structure parameter uncertainty and
between-structure mean uncertainty, but excludes observation noise. It is
named `posterior-epistemic-variance`, not EIG. The branch condition depends
only on posterior class cardinality and the preregistered numerical
certificate; it cannot depend on dataset name, target name, held-out data,
oracle formulas, or observed effectiveness. Raw class-EIG scores, envelopes,
the fixed partition hash, and the selected utility mode are mandatory evidence
fields. Candidate and target actions contain covariates only. The target domain
is fixed before any query response and shared across policies; no held-out
array, response, summary, path, or metadata is available to the score.

P3B.9 preserves that joint score and constrains its decision by a response-free
representativeness guard. Let (A_t) contain the covariates of all currently
observed measurements and let (D^\star) be the fixed registered action-domain
empirical measure. With a positive-definite RBF kernel (k), define the biased
empirical discrepancy

\[
\operatorname{MMD}^2(A_t,D^\star)
=\left\|\frac1{|A_t|}\sum_{x\in A_t}\phi(x)
-\frac1{|D^\star|}\sum_{z\in D^\star}\phi(z)\right\|_{\mathcal H}^2.
\]

Each coordinate is centered and scaled by (D^\star); a constant target
coordinate uses a pooled covariate-only scale. The kernel squared bandwidth is
the median positive pairwise squared distance in the standardized
(D^\star). These operations are invariant to coordinate translation,
nonzero unit scaling, and observation/target/candidate ordering. No label or
task identifier enters them.

For the visible candidate pool (\mathcal A_t), the safe set is

\[
\mathcal S_t=\left\{a\in\mathcal A_t:
\operatorname{MMD}^2(A_t\cup\{a\},D^\star)
\leq \operatorname{MMD}^2(A_t,D^\star)+\tau_{\rm num}\right\},
\]

where (\tau_{\rm num}) is a scale-aware floating-point roundoff allowance.
It is not a fitted hyperparameter. If (\mathcal S_t\neq\varnothing), the
P3B.8 joint-score certificate and its epistemic fallback are evaluated only
over (\mathcal S_t). If (\mathcal S_t=\varnothing), the action minimizing
the augmented MMD is selected. Evidence must separately record safe-set
existence, size, selected MMD, whether the selected action was in the safe set,
whether non-increase held, and whether the explicit minimum-MMD fallback was
used.

P3B.10 preserves this safe set and replaces only the PCPI ranking score. Let
(\mathcal H=\{0.125,0.25,0.5,1.0\}) be the likelihood-power candidates frozen
before this repair. For each (\eta\in\mathcal H), define

\[
J_{t,\eta}(a)=I_{\eta}(C;Y_a\mid H_t)
+\mathbb E_{C,\eta}\!\left[I_{\eta}(Y^\star;Y_a\mid C,H_t)\right].
\]

Every model uses the same observed history, structure bank, frozen basis
preconditioner, initial-frozen structure-to-class map, candidate covariates,
and registered target-domain measure. PCPI ranks with

\[
J_t^{\min}(a)=\min_{\eta\in\mathcal H}J_{t,\eta}(a),
\qquad a\in\mathcal S_t.
\]

The calibrated nominal posterior remains responsible for reported
probabilities, predictive validation, operational-class diagnostics, realized
gain, and all baselines. The ambiguity family is acquisition-only. For each
model the Gauss--Jacobi class-EIG estimate supplies an interval; the robust
leader is certified only when its lower-envelope lower bound exceeds every
eligible competitor's lower-envelope upper bound. If not certified, the
unchanged nominal posterior-epistemic fallback is used. Ties in the reported
least-favorable model are resolved toward the smaller likelihood power.

P3C.1 preserves every P3B.10 role and decision boundary, and adds an
acquisition-only predictive discrepancy envelope. For finite structure member
$S$, let $\widehat r_S^2$ be the residual mean square reconstructed from the
acquired-history sufficient statistics and the current posterior coefficient
mean. With posterior structure mass $p_t(S)$ and the prior mean noise variance
$\sigma_0^2$, define

\[
\widehat\delta_t^2
=\sum_S p_t(S)\max\{0,\widehat r_S^2-\sigma_0^2\}.
\]

Candidate and registered target covariates are standardized by the registered
target domain. If $h^2$ is its median positive squared pairwise distance and
$A_t$ is the acquired design, the extra variance at visible action $a$ is

\[
\delta_t^2(a)=\widehat\delta_t^2
\left(1+\frac{\min_{x\in A_t}\|a-x\|^2}{h^2}\right).
\]

No candidate, validation, or held-out response enters either quantity. For
each Student-t predictive component, location and degrees of freedom are held
fixed while its scale is chosen so its variance increases by
$\delta_t^2(a)$. The class-conditional Gaussian-moment EPIG likewise adds the
candidate and target discrepancy variances to the corresponding marginal
variances, leaving the parameter-induced cross-covariance unchanged. P3C.1
then applies the same finite likelihood-power lower envelope, numerical rank
certificate, representative safe set, and fallback rules as P3B.10.

P3D.1 freezes P3C.1 as negative evidence and tests a decision-layer repair on
an isolated exact fixture. Let the primary utility return to the declared
estimand,

\[
U_t(a)=I(C_0;Y_a\mid H_t),
\]

and let $q_t$ be a response-free reference policy registered over the same
visible candidate set. For a matched random reference, $q_t$ is uniform. Given
valid simultaneous numerical intervals

\[
L_t(a)\le U_t(a)\le R_t(a),
\]

define

\[
L_t(q)=\sum_a q_t(a)L_t(a),\qquad
R_t(q)=\sum_a q_t(a)R_t(a).
\]

The candidate leader maximizes the lower bound, with stable candidate identity
as the tie break. PCPI hands over to this target-seeking action only when

\[
L_t(\widehat a)>R_t(q)+\tau_{\rm num}.
\]

Otherwise it samples from $q_t$. The reference branch is an abstention from an
uncertified target decision, not an EIG failure rebranded as epistemic variance.
If the intervals contain the utilities, the targeted branch has strictly
higher model-based utility than the reference and the fallback branch has
equal reference utility in expectation. This statement is conditional on the
frozen model and valid intervals; it is not a real-world no-harm guarantee
under model misspecification.

P3D.1's exact finite discrete fixture establishes the information identity,
entropy capacity bound, reference aggregation, interval handover,
zero-capacity fallback, stable sampling, permutation invariance, and
fail-closed validation. That controlled Gate passed 14/14 decisions at clean
source commit `5d71f588398daac3a7c8d982ec3eac0b5834d73c` and supports only the
conditional model-relative handover proposition.

P3D.2 supplies actionwise bounds without using the Gauss--Jacobi fine/coarse
diagnostic. Let `Q_a` be the deterministic response quantizer frozen by the
registered probability levels. Data processing gives

\[
L_t(a)=I(C_0;Q_a(Y_a)\mid H_t)\le U_t(a).
\]

For the finite Student-t mixture, Gaussian maximum entropy and concavity of
differential entropy give

\[
U_t(a)\le R_t(a)=\min\left\{H(C_0\mid H_t),
\frac12\log(2\pi e\operatorname{Var}(Y_a))
-\sum_s p_t(s)h(t_{\nu_s,\sigma_s(a)})\right\}.
\]

These inequalities hold in exact arithmetic. The implementation evaluates
Student-t CDFs and entropy terms numerically and expands the interval by a
frozen `1e-10` tolerance; it does not use verified interval arithmetic. The
bounds contain independent adaptive-quadrature class-EIG values on continuous
correctness fixtures and are invariant to positive affine response changes.
Implementation commit `cd05e1a` integrates this rule into a real-only runner
without importing P3B/P3C EPIG, MMD, maximin, discrepancy, or epistemic
fallback terms. No measured P3D.2 result exists yet, so the integration adds no
efficacy evidence.

## 9. Terminal decision

The primary decision is a predictive class. For a preregistered class loss,

\[
\widehat c_T=\arg\min_{\widehat c}
\mathbb E_{c\sim P_T}[L(c,\widehat c)].
\]

The default primary loss is 0–1 loss, giving the maximum-posterior-mass class.
Predictive risk on real validation/confirmation data is a secondary external-validity
measure and cannot retroactively change the posterior target.

## 10. Motif target invariance

If P4 is retained, memory changes proposals only:

\[
q_\lambda(z'\mid z)=(1-\lambda)q_{\mathrm{base}}(z'\mid z)
+\lambda q_{\mathrm{memory}}(z'\mid z).
\]

The reliability gate controls \(\lambda\), never \(\pi_t\). Base support remains
positive, forward/reverse probabilities are computable, and MH/RJ correction is
mandatory. Target-task results cannot update the gate or reusable memory during the
same target experiment.

## 11. Frozen claim boundary

This contract supports the statement: “We formulate symbolic discovery as sequential
Bayesian discrimination among operational predictive-equivalence classes.” The
original finite-bank P2A numerical sub-Gate passed on registered real data with
a genealogy warning; the repaired P2A.1 and P2B correctness Gates subsequently
passed on controlled exact references. P3A.2 supports numerical class-EIG
correctness. P3B.2--P3B.4 are bounded real-development evidence and cannot
support superiority. P3B.5 real efficacy evidence is invalid because its
predictive coordinates differed from its fitted posterior coordinates. P3B.6
repairs that contract and passes its protocol Gate, but its fixed one-SD class
resolution leaves the class variable largely degenerate and does not pass the
efficacy Gate. P3B.7 introduces and validates the budget-resolved primary class
definition, but its returned 96/96 real run fails the unchanged efficacy Gate.
P3B.8 adds the response-free joint class--predictive information target and
passes its controlled diagnostic, while its returned real run remains negative.
P3B.9 and P3B.10 likewise pass their controlled decision-rule diagnostics but
return protocol-valid negative real-development evidence. P3C.1 activates its
scalar discrepancy profile on the Gas family yet still returns
`REAL_ADVANTAGE_NOT_DEMONSTRATED`; on CCPP its profile is effectively zero and
frozen-class gain versus random is significantly negative. This identifies the
P3C.1 discrepancy envelope as an insufficient repair, not a protocol failure.
Motif invariance, open-grammar superiority, physical intervention, VED
discovery, and held-out confirmation remain outside the established evidence
boundary.
