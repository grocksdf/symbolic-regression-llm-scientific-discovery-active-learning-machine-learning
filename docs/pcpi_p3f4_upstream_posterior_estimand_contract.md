# P3F.4 upstream posterior, estimand, and resident-SMC contract

Decision: **CONTRACT-ONLY RESTART. NO P3F.4 INFERENCE IMPLEMENTATION,
CONFIRMATORY FREEZE, PREDICTIVE CALIBRATION, REAL DATA, ACQUISITION, OR
HELD-OUT EXECUTION IS AUTHORIZED.**

Status: **G0--G6 PROOF REVIEW COMPLETE; CURRENT INSTANTIATION NO-GO.** See
pcpi_p3f4_g0_g6_proof_package_20260819.md. G4 and G5 fail for the current
size-three reference and resident budget, so this document does not authorize
implementation.

This document is the upstream response to the final negative P3F.3-VR.8
development result. It does not reinterpret any P3F.3 result as a pass and it
does not define VR.9. Its purpose is to separate the scientific estimand, the
countably open posterior, finite-reference truncation, Monte Carlo
approximation, numerical functional evaluation, and downstream decision error
before another inference mechanism is written.

The source baseline reviewed for this contract is GitHub `main` commit
`ffe7239955d9083a7ad6ef878c3213c602027aad`. The frozen VR.8 archive completed
24/24 runs and has SHA-256
`16bb9ab7e007fd98614ebb3152f929501db75655882958a1c256770ada6d235a`.
It established source identity, matched budgets, proposal invariance, exact
terminal factorization, and bitwise-identical resident paths. It nevertheless
failed five registered development decisions. It is final negative evidence.

## 1. Upstream audit findings

### 1.1 The P3F.3 experiments did not approximate the full open posterior

The registered grammar prior is countably open, but every P3F.3 fidelity and
variance-reduction comparison used `maximum_nodes=3`. The particle engine
therefore targeted

\[
  \pi_L(dz\mid D)=\pi_\infty(dz\mid D, |S|\le L),\qquad L=3,
\]

not the complete posterior \(\pi_\infty\). The reported value
\(\rho^L=0.4^3=0.064\) is the **prior** probability of the omitted size tail.
It is not

\[
  \tau_L(D)=\pi_\infty(|S|>L\mid D),
\]

which is the quantity that controls posterior-functional truncation error.
Data can increase or decrease the tail odds, so prior tail mass alone cannot
certify any full-posterior approximation claim.

The current finite reference also normalizes the grammar prior conditional on
\(|S|\le L\). That is correct for checking \(\pi_L\), but the resulting
normalizing constant is not the finite contribution to the full evidence
unless the slice prior normalization is explicitly undone.

### 1.2 The P3F.3 class variable is not the declared scientific estimand

`OpenTargetParticleResult.equivalence_class_posterior` aggregates exact
integer-polynomial identities. The primary PCPI estimand is the initially
frozen operational predictive-equivalence class. Exact algebraic identity is a
useful correctness diagnostic, but it is not the same random variable as
finite-budget predictive indistinguishability.

The existing complete-link operational partition is defined on a finite
enumerated bank. It is not automatically support-extension invariant: adding
new expressions from an open grammar can alter a clustering or leave the class
of a new expression undefined. A countably open posterior therefore needs a
class map defined for every supported state before the posterior is sampled.

### 1.3 P3F.3 optimized an estimator component that was not dominant

VR.8 integrated the final accept/reject branch and final potential while
holding the resident population fixed. It reduced descriptive mean CDF and
density error on the frozen development bank, but worsened the registered
worst raw-AST, density, and CDF errors. Since resident particles, ESS,
resampling, genealogy, and evidence paths were identical, the remaining error
was already present in the preterminal resident empirical measure. No further
terminal estimator can repair that component.

### 1.4 The old fidelity vector was not derived from downstream Bayes risk

Raw-AST sup-norm error, exact-polynomial class error, pointwise density error,
pointwise CDF error, and log-evidence error are useful diagnostics, but they do
not have equal relevance to the declared decision target. Requiring a
candidate to dominate a comparator for every metric and every fixture is a
mechanism-superiority test, not a certificate that the posterior is accurate
enough for PCPI's class decision or acquisition rule.

P3F.4 must instead derive computational tolerances from a bounded scientific
loss and acquisition regret. Old P3F.3 envelopes remain frozen and negative;
decision-derived P3F.4 tolerances cannot retroactively change them.

### 1.5 Current frontier methods are not drop-in target repairs

- [Bayesian Symbolic Regression via Posterior Sampling](https://arxiv.org/abs/2512.10849)
  and its [NASA PySIPS implementation](https://github.com/nasa/pysips) support
  population sampling and adaptive tempering as engineering directions. The
  published SMC-SR algorithm uses asymmetric mutation/crossover proposals but
  displays an acceptance rule without the complete forward/reverse proposal
  ratio. Those proposals are inadmissible for PCPI unless their actual filtered
  probability is evaluated and corrected.
- [VaSST](https://arxiv.org/abs/2602.23561) obtains scalability through a
  continuous soft-tree variational family and randomized hard-tree recovery.
  It is an approximate target family, not an exact replacement for the frozen
  discrete posterior. A VaSST distribution may later be an auditable proposal
  only if its probability for every emitted hard tree is available for exact
  correction.
- [Probabilistic symbolic forests](https://arxiv.org/abs/2509.19710) provide a
  regularized tree prior and posterior-concentration theory, but their ensemble
  target differs from PCPI's single-expression-plus-discrepancy target. The
  prior and proof strategy are relevant; the posterior cannot be silently
  substituted.
- [Controlled SMC](https://arxiv.org/abs/1708.08396) and
  [Target-Aware Bayesian Inference](https://www.jmlr.org/papers/volume21/19-102/19-102.pdf)
  can reduce error for registered functionals, but control or proposal learning
  must remain exactly corrected and must not define the posterior.

### 1.6 Repository evidence for the resident-family diagnosis

| Audited source | Registered behavior | Consequence |
|---|---|---|
| `configs/p3f_2_open_target_correctness.json` | `reference_slice_maximum_nodes=3` | Exact truth is the conditional size-three slice. |
| P3F.3 fidelity, mechanism, confirmatory, and VR.1--VR.8 configs | `maximum_nodes=3`; VR runs use `complete-uniform` | Every reported fidelity comparison is finite-slice. |
| `hypothesis_mvp/pcpi/open_target/particle.py` | `maximum_nodes=None` is representable, but finite cutoffs divide the prior by `slice_mass`; complete-uniform rejects open support | The API can name an open chain, but the tested resident family is conditional and finite. |
| `hypothesis_mvp/pcpi/open_target/rjmcmc.py` | Proposal kinds are only `complete-uniform` and `prior-independence` | No implemented local typed birth/death open-support kernel has a correctness certificate. |
| `hypothesis_mvp/pcpi/open_target/posterior.py` | Reference operation is `condition-on-node-count-at-most-N` | Its normalized evidence cannot be used as a full-evidence slice contribution without undoing the conditional-prior factor. |
| `hypothesis_mvp/pcpi/open_target/grammar.py` | `tail_mass` is the geometric grammar-prior tail; equivalence aggregation uses exact polynomial identifiers | Neither field is the posterior tail or the operational predictive class. |

Thus the phrase “scalable open-target particle approximation” currently
describes an intended architecture, not a demonstrated full-open resident SMC
family. This is a claim-boundary correction, not a criticism of the finite-slice
correctness proofs.

## 2. Objects that must remain distinct

P3F.4 registers five different mathematical objects.

1. **Full generative target** \(\pi_\infty\): the ordinary Bayesian posterior
   on the countably open grammar.
2. **Finite exact reference** \(\pi_L\): the full posterior conditioned on a
   registered size slice, used only for exact checks.
3. **Operational pushforward** \(C_\star\#\pi_\infty\): the posterior law of
   the scientific class variable.
4. **Resident particle approximation** \(\widehat\pi_N\): the random empirical
   measure propagated by SMC.
5. **Numerical functional approximation** \(\widehat T(\widehat\pi_N)\): CDF,
   predictive, entropy, or acquisition computations performed on the empirical
   measure.

No field named `posterior`, `equivalence_class`, `tail_mass`, `evidence`, or
`calibration` may refer to more than one of these objects without an explicit
qualifier.

## 3. Full posterior target

Let the collapsed discrete state be

\[
  z=(S,d,\lambda),
\]

where continuous expression amplitude, discrepancy coordinates, and noise
variance are integrated under the registered proper prior. For this contract
version, the P3F.2 Gaussian-homoscedastic NIG likelihood and response-free
structure-wise discrepancy remain fixed. Noise-family, measurement-error, and
operator-language expansion are separate target versions; they must not be
mixed into the first P3F.4 inference certificate.

Define

\[
  \gamma_\infty(z;D)
  =p_{\mathcal G}(S)p(d,\lambda\mid S)m(D\mid z),
  \qquad
  Z_\infty(D)=\sum_z\gamma_\infty(z;D),
\]

and

\[
  \pi_\infty(z\mid D)=\gamma_\infty(z;D)/Z_\infty(D).
\]

The primary target is always \(\pi_\infty\). A computational cutoff is not a
target parameter.

For an exact slice \(A_L=\{z:|S|\le L\}\), define the **unnormalized full-target
slice contribution**

\[
  Z_L(D)=\sum_{z\in A_L}\gamma_\infty(z;D)
\]

before defining the conditional reference

\[
  \pi_L(z\mid D)=
  \frac{\gamma_\infty(z;D)\mathbf 1\{z\in A_L\}}{Z_L(D)}.
\]

This prevents a conditional-slice evidence from being mistaken for full
evidence.

## 4. Posterior-tail certificate

The posterior tail is

\[
  \tau_L(D)=1-\frac{Z_L(D)}{Z_\infty(D)}
  =\frac{Z_{>L}(D)}{Z_L(D)+Z_{>L}(D)}.
\]

P3F.4 must produce a response-valid upper bound
\(Z_{>L}(D)\le U_{>L}(D)\). An admissible construction may combine exact shell
enumeration with an analytic marginal-likelihood envelope for the remaining
grammar prior. The resulting certificate is

\[
  \tau_L(D)\le
  \bar\tau_L(D)=\frac{U_{>L}(D)}{Z_L(D)+U_{>L}(D)}.
\]

If no finite, non-vacuous \(U_{>L}\) is available, full-open fidelity fails
closed. The prior tail \(\rho^L\), empirical absence of large trees, or agreement
between two finite cutoffs is descriptive only.

For any registered \(0\le\varphi\le1\),

\[
  |\pi_\infty(\varphi)-\pi_L(\varphi)|\le\tau_L(D)
  \le\bar\tau_L(D).
\]

The same bound applies to every operational-class probability and predictive
CDF value. Pointwise density is not automatically bounded by total variation;
it therefore remains secondary unless a finite density envelope is proved.
Integrated bounded scores or CDF error are the primary predictive fidelity
objects.

## 5. Support-extension-invariant operational estimand

P3F.4 replaces finite-bank clustering with a deterministic map defined on the
entire target support. This is a new estimand version and must not be presented
as the unchanged P3B class variable.

Freeze before any acquisition response:

- a visible action grid \(\mathcal A_0=(a_1,\ldots,a_I)\);
- a response-threshold grid \(\mathcal R_0=(r_1,\ldots,r_J)\) derived only from
  the initial visible role and registered physical scale;
- a bin count \(K_B\) derived from the future measurement budget and a
  decision-regret requirement, not from an inference result; and
- a stable bin-boundary convention.

For every supported collapsed state \(z\), define its initial predictive-CDF
signature

\[
  \Psi_0(z)=
  \big(F_z(r_j\mid a_i,H_0)\big)_{i=1:I,j=1:J}
  \in[0,1]^{IJ}.
\]

With \(h_B=1/K_B\), define coordinate bins

\[
  q_B(u)=\min\{K_B-1,\lfloor u/h_B\rfloor\}
\]

and the class map

\[
  C_\star(z)=\big(q_B(\Psi_{0,k}(z))\big)_{k=1}^{IJ}.
\]

This map has at most \(K_B^{IJ}\) values, is deterministic, transitive, defined
for unseen structures, and invariant to the order in which support is
discovered. Members of one class differ by at most \(h_B\) at every registered
predictive-CDF coordinate; equality is possible only at the closed endpoint of
the final bin. Exact polynomial identity is retained as a separate algebraic
diagnostic.

The bin count, action grid, threshold grid, and initial-history hash are part of
the estimand identity. Boundary sensitivity is reported using a separately
preregistered half-bin shift but cannot replace the primary map.

\(C_\star\) is exact for this finite, grid-restricted operational estimand. It
does not by itself establish predictive equivalence between action or response
grid points. Any continuum-domain scientific claim must separately register a
full predictive signature and a response-free modulus-of-continuity bound that
turns action-grid and threshold-grid spacing into a deterministic coarsening
error. Without that bound, the claim remains explicitly grid-restricted.

The definition also uses the mathematical CDF, not a silently rounded floating
point value. If a certified numerical interval for any coordinate crosses a bin
boundary, the implementation returns `boundary-uncertain` and propagates class
probability bounds. It may not assign the coordinate to the nearest bin.
Estimand coarsening, boundary uncertainty, posterior tail, particle error, and
functional numerical error are five distinct quantities.

## 6. Error decomposition

Two decompositions are registered and must not be mixed.

For a finite-slice correctness implementation
\(\widehat\pi_{N,L}\) that actually targets \(\pi_L\), and a bounded registered
functional \(\varphi\),

\[
\begin{aligned}
|\pi_\infty(\varphi)-\widehat T(\widehat\pi_{N,L})|
\le{}&
\underbrace{\bar\tau_L}_{\text{posterior tail}}
\\&+
\underbrace{|\pi_L(\varphi)
-E\widehat\pi_{N,L}(\varphi)|}_{\text{finite-}N\text{ bias}}
\\&+
\underbrace{|E\widehat\pi_{N,L}(\varphi)
-\widehat\pi_{N,L}(\varphi)|}_{\text{island randomness}}
\\&+
\underbrace{|\widehat\pi_{N,L}(\varphi)
-\widehat T(\widehat\pi_{N,L})|}_{\text{functional numerics}}.
\end{aligned}
\]

For the production open-support implementation
\(\widehat\pi_N^\infty\), which targets \(\pi_\infty\) rather than \(\pi_L\),
the primary decomposition is instead

\[
\begin{aligned}
|\pi_\infty(\varphi)-\widehat T(\widehat\pi_N^\infty)|
\le{}&
\underbrace{|\pi_\infty(\varphi)
-E\widehat\pi_N^\infty(\varphi)|}_{\text{finite-}N\text{ bias}}
\\&+
\underbrace{|E\widehat\pi_N^\infty(\varphi)
-\widehat\pi_N^\infty(\varphi)|}_{\text{island randomness}}
\\&+
\underbrace{|\widehat\pi_N^\infty(\varphi)
-\widehat T(\widehat\pi_N^\infty)|}_{\text{functional numerics}}.
\end{aligned}
\]

The exact slice supplies the additional conservative check

\[
|\pi_\infty(\varphi)-\widehat T(\widehat\pi_N^\infty)|
\le
\bar\tau_L
+|\pi_L(\varphi)-\widehat T(\widehat\pi_N^\infty)|.
\]

The second term here is **reference disagreement**, not pure particle error:
it may include correctly sampled mass outside \(A_L\). A field or Gate may not
label it otherwise. Generative-model misspecification is also not hidden
inside any computational bound. It is evaluated later by predictive
calibration under a separately frozen target version.

The primary computational distance is the total variation error of the
operational pushforward and the registered joint law required by acquisition,
not raw-AST max error:

\[
  \delta_C=
  \|C_\star\#\pi_\infty-C_\star\#\widehat\pi_N\|_{\rm TV}.
\]

Raw-AST, exact-polynomial class, ESS, genealogy, acceptance, and evidence
remain necessary diagnostics. They do not substitute for \(\delta_C\).

## 7. Decision-derived tolerances

For default 0--1 class loss, the Bayes risk is

\[
  R(p)=1-\max_c p(c).
\]

If \(\|p-q\|_{\rm TV}\le\delta_C\), then

\[
  |R(p)-R(q)|\le\delta_C,
\]

and the decision selected under \(q\) has posterior regret under \(p\) at most
\(2\delta_C\). Thus the permissible posterior error must be obtained from a
preregistered regret budget \(r_\star\):

\[
  \delta_C\le r_\star/2.
\]

For a bounded acquisition utility \(0\le U(a,z)\le U_{\max}\), a uniform
posterior TV error \(\delta\) gives

\[
  \sup_a|E_\pi U(a,Z)-E_{\widehat\pi}U(a,Z)|
  \le U_{\max}\delta,
\]

and plug-in action regret at most \(2U_{\max}\delta\). Entropy/EIG decisions
must additionally use a finite-class entropy continuity envelope. If the
leader's lower utility bound does not exceed every competitor's upper bound,
the inference layer returns `uncertified` and the decision layer uses the
already registered reference/abstention rule.

This replaces arbitrary accuracy thresholds with a decision certificate. It
does not revise any old P3F.3 threshold.

## 8. Resident SMC family contract

The first admissible resident family is `full-open-smc-v1`. It is a new
algorithm family, not VR.9.

### 8.1 Support and proposals

- Initial particles are sampled from the exact countably open grammar prior;
  `maximum_nodes=None` is part of the production identity.
- A finite-support complete-uniform proposal is forbidden in the production
  open chain.
- The proposal mixture contains a strictly positive exact prior-independence
  component and, only after proof, reversible typed birth/death/replacement
  components.
- Every filter, canonicalization, invalid-tree rejection, and operator rule is
  included in the forward and reverse proposal probabilities.
- Learned, LLM, evolutionary, flow, or variational proposals may enter only as
  extra mixture components with evaluable post-filtering probability. Their
  removal must not change target support.

### 8.2 Feynman--Kac path

- Every intermediate target has the same full-open support and terminates at
  the ordinary Bayesian posterior.
- The new-observation likelihood may be fractionally bridged. For normalized
  incremental potential \(G\), the population quantity

  \[
    r_{\rm ESS}=\frac{\pi(G)^2}{\pi(G^2)}
    =\frac{1}{1+\chi^2(\pi'\|\pi)}
  \]

  defines the adjacent \(L_2\)/chi-square path condition. A bridge is accepted
  only from the registered analytic normalizer-interval lower certificate, or
  a separately proved finite-particle lower confidence bound on
  \(r_{\rm ESS}\), not from the raw plug-in CESS value alone.
- No forced beta increment, fixed-step completion rule, generalized likelihood
  power, or response-dependent target change is allowed.
- The exact finite slice must report the true adjacent \(L_2\) ratios so that
  the adaptive estimate can be checked. Recent finite-sample analysis shows
  that SMC complexity depends jointly on adjacent \(L_2\) distance and kernel
  mixing, and that ordinary data tempering need not automatically satisfy the
  required path conditions
  ([Marion, Mathews, and Schmidler](https://arxiv.org/abs/1807.01346)).

### 8.3 Resampling and mixing

- Resampling is unbiased and registered; ordinary ESS, conditional ESS,
  maximum normalized weight, and \(\infty\)-ESS are all recorded.
- Rejuvenation kernels leave the current target invariant. Finite-slice
  transition matrices must pass row normalization, detailed balance,
  stationarity, irreducibility, and a mixing/spectral-gap certificate.
- Genealogy is a failure diagnostic, not an accuracy proof.
- SMC normalizing-constant estimates and empirical posterior expectations are
  distinct estimators. Unbiasedness of the former must not be transferred to
  the latter.

Direct divergence control and \(\infty\)-ESS are motivated by
[Huggins and Roy](https://arxiv.org/abs/1503.00966); the general target,
proposal, twisting, and normalizing-constant separation follows standard SMC
sampler theory summarized in
[Elements of Sequential Monte Carlo](https://arxiv.org/abs/1903.04797).

### 8.4 Replication and uncertainty

The evidence unit is an independent SMC island, not an individual particle or
seed-picked run. A preregistered total evaluation budget is divided among
independent islands. For bounded functionals, concentration across islands
controls only
\(|E\widehat\pi_N(\varphi)-\widehat\pi_N(\varphi)|\). It does not remove the
finite-\(N\) bias
\(|\pi(\varphi)-E\widehat\pi_N(\varphi)|\), which requires a separate
nonasymptotic bound from the registered path and mixing assumptions. The two
bounds are added, never conflated.

Scalar functional estimates include a simultaneous high-probability interval;
vector class/predictive estimates use a registered simultaneous norm bound.
Means and favorable individual seeds are descriptive only.

The number of islands, per-island population, aggregation rule, confidence
level, and total target evaluations are frozen before responses. A larger total
budget cannot be introduced after a failure.

## 9. Evidence lanes

### Lane A: exact finite references

Use multiple response-free fixtures and an increasing registered sequence of
enumerable size slices. For each slice:

1. compute unnormalized full-target slice evidence;
2. prove proposal invariance and target stationarity;
3. compare full class pushforward, predictive CDF signature, and acquisition
   joint law;
4. compute exact adjacent path \(L_2\) ratios and finite-kernel mixing
   quantities;
5. verify simultaneous Monte Carlo coverage rather than only average or
   worst fixed-seed error; and
6. verify the decision-regret certificate.

### Lane B: complete open support

Without access to real or held-out responses:

1. demonstrate nonzero sampling and reversible movement across registered size
   shells;
2. report posterior size-shell mass without truncating it away;
3. produce the posterior-tail upper certificate;
4. compare at least two target-correct proposal mixtures at matched total
   budget;
5. verify that operational class identities do not change when new structures
   are encountered; and
6. fail closed if the tail, path, or decision error cannot be certified.

Lane A passing alone proves finite-reference correctness. Lane B plus a
non-vacuous tail and decision certificate is required for a scalable
open-posterior fidelity claim.

## 10. Gates and proof obligations before code

No P3F.4 production implementation may begin until the following proof package
is reviewed:

1. **G0 Estimand:** prove that \(C_\star\) is measurable, finite,
   support-extension invariant, and uses no future response; freeze whether the
   scientific claim is grid-restricted or supply the continuum coarsening
   bound; define fail-closed numerical boundary handling.
2. **G1 Target:** prove proper normalization of the complete grammar/latent
   prior and identify the unnormalized slice contribution to full evidence.
3. **G2 Tail:** derive a computable, non-vacuous \(U_{>L}(D)\) and its
   functional-error consequences.
4. **G3 Open kernel:** write exact forward/reverse probabilities and a
   finite-slice detailed-balance proof for every proposed move.
5. **G4 Path:** define the population relative-ESS/\(L_2\) condition, its
   analytic or separately proved finite-particle lower certificate, and
   fail-closed behavior without a forced terminal step.
6. **G5 Decision:** derive the tail, finite-\(N\) bias, island-randomness, and
   numerical error allocation from a frozen Bayes-risk/acquisition-regret
   budget.
7. **G6 Roles:** prove that the target priors cannot access responses; the
   initial class map may use only frozen \(H_0\) and selection-visible
   covariates; and no object may use mechanism-development outcomes, future
   acquisition responses, real validation, or held-out state.

Passing these obligations authorizes only a correctness implementation and
response-free fixtures. It does not authorize a confirmatory freeze.

## 11. No-go conditions

Stop this stage if any of the following occurs:

- posterior tail is replaced by prior tail;
- the exact reference renormalizes a slice without retaining its full-evidence
  contribution;
- an open-support class label depends on the set or order of sampled trees;
- a grid-restricted class is described as continuum predictive equivalence
  without a registered coarsening bound;
- numerical CDF error crossing a class-bin boundary is silently rounded;
- exact polynomial identity is relabelled as operational predictive
  equivalence;
- an SMC proposal cannot evaluate its true filtered forward/reverse
  probability;
- a variational, LLM, genetic, or controlled sampler defines the target rather
  than an exactly corrected proposal;
- ESS or genealogy is used as a posterior-distance certificate by itself;
- a tolerance is selected from P3F.3 failures rather than a decision-regret
  budget;
- a full-open claim is based only on finite-slice agreement; or
- code is written before G0--G6 are complete.

## 12. Consequence for the AISTATS claim

P3F.3-VR.1 through VR.8 establish a rigorous negative boundary: target-correct
terminal and resident variance-reduction variants did not yield stable
finite-budget dominance under their frozen Gates. They do not establish that
the complete open posterior was approximated.

The next defensible technical claim, if P3F.4 succeeds, is narrower and more
important:

> a countably open symbolic posterior with a support-extension-invariant
> operational estimand, explicit posterior-tail error, and decision-derived
> finite-sample SMC fidelity certificate.

Until that claim is proved, predictive calibration, real acquisition,
held-out confirmation, efficacy, and discovery superiority remain blocked.
