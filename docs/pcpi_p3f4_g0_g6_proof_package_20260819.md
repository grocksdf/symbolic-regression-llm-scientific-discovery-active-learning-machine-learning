# P3F.4 G0--G6 mathematical proof package

Decision: **PROOF REVIEW COMPLETE; CURRENT P3F.4 INSTANTIATION IS NO-GO.**

This package discharges every requested review item by either a proof or a
formal counter-certificate. It does not reinterpret a failed Gate as
incomplete work. G4 and G5 fail for the current size-three reference and
resident budget, so no inference implementation, confirmatory freeze,
predictive calibration, real-data run, acquisition, or held-out execution is
authorized.

The source baseline is GitHub main commit
ffe7239955d9083a7ad6ef878c3213c602027aad. The frozen P3F.3-VR.8 archive has
SHA-256
16bb9ab7e007fd98614ebb3152f929501db75655882958a1c256770ada6d235a
and remains final negative evidence.

## 1. Gate dispositions

| Gate | Disposition | Mathematical result | Current-instantiation result |
|---|---|---|---|
| G0 Estimand | PROVED | A finite grid-restricted CDF-signature map is measurable, transitive, and support-extension invariant after Amendments A1--A2 below. | It is a new estimand and cannot be presented as the old complete-link class. |
| G1 Target | PROVED | The raw-AST grammar/component/NIG prior and full evidence define a proper countably open posterior. The conditional-slice evidence conversion is exact. | The posterior is grammar-representation dependent; that limitation must be named. |
| G2 Tail | PROVED | A finite response-valid marginal-likelihood envelope gives a computable posterior-tail bound for every finite slice. | At L=3 the bound is 0.9238--0.9996 on the frozen VR.8 fixtures and is decision-useless. |
| G3 Open kernel | PROVED | The exact prior-independence MH kernel is reversible, irreducible, aperiodic, and uniformly ergodic on the open support. | Only the prior-independence base kernel is admitted. No local move is authorized. |
| G4 Path | NO-GO | A deterministic lower certificate for population relative ESS is derived from finite-slice normalizer intervals. | The certified lower bound falls to 4.50e-8; the registered 0.8 path floor cannot be certified at L=3. |
| G5 Decision | NO-GO | Explicit tail, terminal-mixing, independent-island, MAP, and quantized-EIG error bounds are derived. | Tail error alone makes every nontrivial class/action certificate vacuous under the current reference and budget. |
| G6 Roles | PROVED | A response-access noninterference table is complete after distinguishing initial history from mechanism-development and future responses. | Future implementation must enforce these capabilities; the proof does not authorize that implementation. |

Because G4 and G5 are NO-GO, the package-wide disposition is NO-GO even
though the mathematical target and base kernel are valid.

## 2. Normative amendments required by the proof

### A1. Initial history is legitimate estimand input

The operational class is an initial-posterior object. It may depend on the
registered initial history

\[
  H_0=(X_0,Y_0)
\]

and on selection-visible action covariates. It may not depend on P3F.3/P3F.4
mechanism-development outcomes, future acquisition responses, real validation
responses, or held-out state.

This corrects the overly broad draft phrase that denied the class map all
development responses. Target hyperparameters and the grammar remain
response-independent; the class map is allowed to condition on the scientific
history that defines its estimand.

### A2. The first operational class is grid-restricted

The first P3F.4 scientific claim is explicitly grid-restricted. It does not
claim continuum predictive equivalence.

Freeze:

- future acquisition budget \(B=32\);
- canonical action grid \(\mathcal A_0\), consisting of the sorted
  selection-visible operational actions at \(H_0\);
- response probability levels

  \[
    \mathcal U_0=(0.05,0.15,0.30,0.50,0.70,0.85,0.95);
  \]

- initial response center \(\mu_0\) and positive initial response scale
  \(s_0\), both computed from \(Y_0\) by the frozen initial standardizer; and
- thresholds

  \[
    r_j=\mu_0+s_0\Phi^{-1}(u_j),\qquad u_j\in\mathcal U_0.
  \]

If \(s_0\) is not finite and strictly positive, G0 fails closed. No
result-derived replacement scale is permitted.

Set

\[
  K_B=\lceil\sqrt B\rceil=6,\qquad h_B=1/K_B=1/6.
\]

Then \(h_B\le1/\sqrt B\), matching the already registered one-unit
root-budget predictive resolution in probability units.

### A3. G4 uses an analytic path certificate

Raw plug-in CESS is a diagnostic, not a proof. A bridge is admissible only
when the analytic population-relative-ESS lower bound in Section 7 is at least
the frozen floor \(r_0=0.8\). A separately proved finite-particle confidence
lower bound may replace the analytic bound later, but no unproved plug-in
quantity may do so.

### A4. G5 uses margin certification and reference fallback

There is no arbitrary new posterior-error threshold. Class decisions are
released only when simultaneous probability intervals certify the MAP class.
Targeted acquisition is released only when its lower utility bound exceeds
the registered reference policy's upper bound. Otherwise the decision layer
returns the registered reference/abstention action.

This freezes a zero certified regret requirement relative to the reference
policy while permitting finite computation whenever the scientific margin is
large enough.

## 3. G0 proof: operational estimand

### Definition

Let the collapsed state space be

\[
  \mathcal Z=\bigcup_{\ell\ge1}
  \mathcal S_\ell\times\mathcal D,
\]

where every \(\mathcal S_\ell\) is finite and
\(\mathcal D=\{\mathrm{spike}\}\cup
\{\mathrm{slab},\lambda:\lambda\in\Lambda\}\) is finite.

For every \(z\in\mathcal Z\), define

\[
 \Psi_0(z)=
 \left(
 F_z(r_j\mid a_i,H_0)
 \right)_{i=1:I,j=1:7}.
\]

Let

\[
 q_B(u)=\min\{K_B-1,\lfloor K_Bu\rfloor\},
\]

and

\[
 C_\star(z)=
 \left(q_B(\Psi_{0,k}(z))\right)_{k=1}^{7I}.
\]

### Proposition G0.1: measurability

\(\mathcal Z\) is countable and carries the discrete sigma algebra. Every map
from a discrete measurable space is measurable. Therefore \(\Psi_0\) and
\(C_\star\) are measurable.

### Proposition G0.2: finiteness

Every coordinate of \(C_\star\) lies in
\(\{0,\ldots,K_B-1\}\). Hence

\[
 |\operatorname{range}(C_\star)|
 \le K_B^{7I}<\infty.
\]

### Proposition G0.3: equivalence relation

Define \(z\sim z'\) if and only if \(C_\star(z)=C_\star(z')\). Equality of
finite vectors is reflexive, symmetric, and transitive, so \(\sim\) is an
equivalence relation.

### Proposition G0.4: support-extension invariance

\(C_\star(z)\) depends only on \(z\), \(H_0\), and the frozen grids. It does
not depend on an enumerated bank, sampled population, cluster merge order, or
the presence of another state. Adding any new supported state therefore
cannot change an existing state's label.

### Proposition G0.5: operational diameter

States in the same class occupy the same width-\(h_B\) bin at every
coordinate. Their CDF values differ by at most \(h_B=1/6\), with equality
possible only at the closed endpoint of the final bin. Since

\[
 1/6\le1/\sqrt{32},
\]

the class meets the frozen coordinatewise root-budget resolution.

### Proposition G0.6: response noninterference

\(H_0\) is frozen before acquisition. \(\mathcal A_0\) contains only
selection-visible covariates; \(\mathcal U_0\), \(B\), and \(K_B\) are fixed
constants. Thus no future acquisition, real-validation, held-out, or
mechanism-development response can change \(C_\star\).

### Boundary rule

The mathematical CDF defines the class. If a certified numerical interval for
any coordinate intersects two bins, the state is boundary-uncertain and its
mass is propagated to both admissible class bounds. Nearest-bin rounding is
forbidden.

### Scientific limitation

This proves only grid-restricted predictive equivalence. It does not prove
equivalence between action points or outside the seven response thresholds.
It also does not remove grammar-representation dependence from the posterior
prior mass.

## 4. G1 proof: proper full posterior

### Proposition G1.1: grammar normalization

For node count \(\ell\ge1\),

\[
 p(L=\ell)=(1-\rho)\rho^{\ell-1},\qquad 0<\rho<1.
\]

There are \(N_\ell<\infty\) registered ASTs of size \(\ell\), each with
conditional probability \(1/N_\ell\). Therefore

\[
\sum_{S\in\mathcal S}p_{\mathcal G}(S)
=\sum_{\ell=1}^\infty
N_\ell\frac{(1-\rho)\rho^{\ell-1}}{N_\ell}
=1.
\]

### Proposition G1.2: collapsed component normalization

For each structure \(S\), the spike has mass \(1-\pi_d\). Slab components
have masses \(\pi_d p(\lambda)\), and the registered kernel probabilities sum
to one. Consequently

\[
 \sum_{d,\lambda}p(d,\lambda\mid S)
 =(1-\pi_d)+\pi_d\sum_\lambda p(\lambda)=1.
\]

All coefficient/discrepancy Gaussian priors and the inverse-gamma noise prior
are proper because their registered precisions, shape, and scale are strictly
positive.

### Proposition G1.3: uniform marginal-likelihood envelope

Let \(n\) responses be observed. For any finite design matrix, any component
dimension, and the registered Normal--Inverse-Gamma prior with shape \(a_0\)
and scale \(b_0\), the collapsed marginal likelihood is

\[
m_z(D)=
(2\pi)^{-n/2}
\frac{|P_0|^{1/2}}{|P_n|^{1/2}}
\frac{b_0^{a_0}}{b_n^{a_0+n/2}}
\frac{\Gamma(a_0+n/2)}{\Gamma(a_0)}.
\]

Since \(P_n=P_0+X^\top X\succeq P_0\),

\[
 |P_0|^{1/2}/|P_n|^{1/2}\le1.
\]

Completing the square gives

\[
b_n=b_0+\frac12
\min_\theta\{
\|Y-X\theta\|^2+
(\theta-\mu_0)^\top P_0(\theta-\mu_0)
\}\ge b_0.
\]

Therefore

\[
0<m_z(D)\le
M_n:=
(2\pi)^{-n/2}
\frac{\Gamma(a_0+n/2)}{\Gamma(a_0)}
b_0^{-n/2}<\infty.
\]

The bound is independent of AST size, expression values, discrepancy rank,
and kernel state.

### Proposition G1.4: posterior propriety

With \(p_0(z)=p_{\mathcal G}(S)p(d,\lambda\mid S)\),

\[
0<Z_\infty(D)
=\sum_zp_0(z)m_z(D)
\le M_n\sum_zp_0(z)
=M_n<\infty.
\]

Hence

\[
\pi_\infty(z\mid D)=p_0(z)m_z(D)/Z_\infty(D)
\]

is a proper posterior on the complete countably open support.

### Proposition G1.5: exact slice-evidence conversion

The existing finite reference uses

\[
p_L(S)=
\frac{p_{\mathcal G}(S)}
{1-\rho^L}\mathbf1\{|S|\le L\}.
\]

If its reported conditional evidence is \(Z_L^{\rm cond}\), the contribution
of the same states to the full evidence is exactly

\[
 Z_L=(1-\rho^L)Z_L^{\rm cond}.
\]

This factor must be restored before any posterior-tail calculation.

### Target-domain condition

For the registered structure-wise RBF discrepancy to exist for every
structure, the action rows must be finite and pairwise distinct and \(n\ge3\).
Then each RBF kernel is positive definite, the one-column expression design
has rank at most one, and its orthogonal complement has positive dimension.
Inputs outside this domain fail closed.

### Scientific limitation: representation dependence

The posterior is proper but is a raw-AST posterior. Algebraically redundant
representations such as double negations and multiplication by one receive
separate prior mass. Thus \(C_\star\#\pi_\infty\) is grammar-relative and may
change under semantics-preserving grammar rewrites. No representation-
invariant scientific claim is authorized without a later canonical-semantic
prior contract.

## 5. G2 proof: posterior-tail certificate

### Proposition G2.1: basic tail bound

The omitted prior mass after size \(L\) is \(\rho^L\). By Proposition G1.3,

\[
Z_{>L}(D)
=\sum_{|S|>L,d,\lambda}p_0(z)m_z(D)
\le M_n\rho^L
=:U_{>L}(D).
\]

Therefore

\[
\tau_L(D)
\le
\bar\tau_L(D)
=
\frac{M_n\rho^L}
{Z_L(D)+M_n\rho^L}.
\]

This is a posterior-tail bound, not a relabelled prior tail.

### Proposition G2.2: exact-shell refinement

For any \(R>L\), let \(Z_\ell(D)\) be the exact unnormalized full-target
contribution of shell \(|S|=\ell\). Then

\[
U_{>L}^{(R)}(D)
=\sum_{\ell=L+1}^{R}Z_\ell(D)+M_n\rho^R
\]

is also valid and is never worse than replacing all shells after \(L\) by the
basic envelope when their exact contributions are smaller than that envelope.

### Proposition G2.3: bounded-functional consequence

Because

\[
\pi_\infty=(1-\tau_L)\pi_L+\tau_L\pi_{>L},
\]

for every \(0\le\varphi\le1\),

\[
|\pi_\infty(\varphi)-\pi_L(\varphi)|
\le\tau_L\le\bar\tau_L.
\]

This includes class probabilities and predictive CDF coordinates. It does not
give a pointwise density bound.

### Frozen VR.8 numerical counter-certificate

For \(n=8\), \(a_0=3\), \(b_0=0.08\), \(\rho=0.4\), and \(L=3\),

\[
M_8=5639.272478769489,\qquad M_8\rho^3=360.91343864124735.
\]

The archived exact evidences are conditional-slice evidences. Restoring the
factor \(1-\rho^3=0.936\) gives:

| Fixture | \(Z_3\) | \(U_{>3}\) | \(\bar\tau_3\) |
|---|---:|---:|---:|
| AC | 0.1467437810 | 360.9134386412 | 0.9995935753 |
| AD | 0.3434053165 | 360.9134386412 | 0.9990494151 |
| AE | 29.7661541284 | 360.9134386412 | 0.9238092937 |

The theorem is valid, but the current certificate is practically vacuous.
For example, using only \(Z_3\) as a lower evidence bound, a one-percent tail
certificate requires at least sizes \(L=17,16,11\) for AC, AD, and AE,
respectively. The registered grammar contains approximately 5.92 billion raw
ASTs through size 17. Exact raw-AST enumeration is therefore not a scalable
route to that certificate.

G2 is mathematically PROVED, while the current L=3 fidelity claim is blocked.

## 6. G3 proof: base open reversible kernel

### Admitted proposal

Only the exact prior-independence proposal is admitted:

\[
q_0(z'\mid z)=p_0(z').
\]

It includes self proposals and assigns positive probability to every supported
collapsed state. Complete-uniform is excluded because the support is infinite.
Local typed birth/death/replacement, learned, variational, LLM, genetic, and
flow proposals remain excluded until separate exact post-filtering
probabilities are proved.

At a bridge target

\[
\pi_j(z)=\gamma_j(z)/Z_j
=p_0(z)m_j(z)/Z_j,
\]

use

\[
\alpha_j(z,z')
=\min\left\{
1,\frac{\gamma_j(z')p_0(z)}
{\gamma_j(z)p_0(z')}
\right\}
=\min\{1,m_j(z')/m_j(z)\}.
\]

### Proposition G3.1: detailed balance

For \(z\ne z'\),

\[
\begin{aligned}
\pi_j(z)p_0(z')\alpha_j(z,z')
&=\frac1{Z_j}
\min\{p_0(z)m_j(z)p_0(z'),
p_0(z')m_j(z')p_0(z)\}\\
&=\pi_j(z')p_0(z)\alpha_j(z',z).
\end{aligned}
\]

The diagonal mass is the rejection plus self-proposal probability, so the
complete kernel is reversible and stationary for \(\pi_j\).

### Proposition G3.2: irreducibility and aperiodicity

Every target state has \(p_0(z')>0\), \(m_j(z')>0\), and hence a positive
one-step proposal and acceptance probability from every source. The kernel is
\(\pi_j\)-irreducible. Since \(q_0(z\mid z)=p_0(z)>0\), every state has a
positive self-transition probability; the chain is aperiodic.

### Proposition G3.3: uniform-ergodicity certificate

Let \(M_j\) be any uniform upper bound on \(m_j(z)\). Then

\[
\frac{\pi_j(z)}{p_0(z)}
=\frac{m_j(z)}{Z_j}
\le\frac{M_j}{Z_j}.
\]

The independence-MH minorization therefore gives

\[
K_j(z,A)\ge\epsilon_j\pi_j(A),
\qquad
\epsilon_j=Z_j/M_j.
\]

Using a finite slice,

\[
\epsilon_j\ge\underline\epsilon_j
:=Z_{L,j}/M_j>0,
\]

and consequently

\[
\|\mu K_j^m-\pi_j\|_{\rm TV}
\le(1-\underline\epsilon_j)^m
\]

for every initial law \(\mu\).

This is an exact open-support mixing proof, although the lower bound may be
too small for practical use. At the final VR.8 targets the L=3 lower bounds
are:

| Fixture | \(\underline\epsilon_T=Z_3/M_8\) | Steps for TV mixing term \(\le0.01\) |
|---|---:|---:|
| AC | 2.60218e-5 | 176,972 |
| AD | 6.08953e-5 | 75,623 |
| AE | 5.27837e-3 | 871 |

The current one-step terminal rejuvenation budget is not a mixing
certificate. G3 nevertheless proves the base kernel itself.

Primary theoretical reference:
[Mengersen and Tweedie, Rates of convergence of the Hastings and Metropolis algorithms](https://doi.org/10.1214/aos/1033066201).

## 7. G4 proof and current NO-GO: Feynman--Kac path

### Fractional targets

At observation \(t\), let \(m_{t,\beta}(z)\) be the collapsed marginal with
the first \(t-1\) rows at power one and row \(t\) at power
\(\beta\in[0,1]\). Define

\[
\gamma_{t,\beta}(z)=p_0(z)m_{t,\beta}(z),
\qquad
Z_{t,\beta}=\sum_z\gamma_{t,\beta}(z).
\]

For any real effective count \(s=t-1+\beta\ge0\), the G1 determinant/scale
argument gives

\[
m_{t,\beta}(z)\le
M_s:=
(2\pi)^{-s/2}
\frac{\Gamma(a_0+s/2)}{\Gamma(a_0)}
b_0^{-s/2}.
\]

Hence

\[
Z_{L,t,\beta}
\le Z_{t,\beta}
\le Z_{L,t,\beta}+M_s\rho^L
=:\overline Z_{t,\beta}.
\]

### Population relative ESS

For a bridge \(\beta<\beta'\), let

\[
G(z)=m_{t,\beta'}(z)/m_{t,\beta}(z).
\]

The population relative ESS is

\[
r_{\rm ESS}
=\frac{\pi_{t,\beta}(G)^2}
{\pi_{t,\beta}(G^2)}.
\]

For each collapsed state, \(m_{t,\beta}\) is the moment-generating integral of
the current likelihood power and is log-convex in \(\beta\). Cauchy--Schwarz
therefore gives

\[
\frac{m_{t,\beta'}(z)^2}{m_{t,\beta}(z)}
\le m_{t,2\beta'-\beta}(z).
\]

It follows that

\[
r_{\rm ESS}
\ge
\frac{Z_{t,\beta'}^2}
{Z_{t,\beta}Z_{t,2\beta'-\beta}}
\ge
\underline r_{t}(\beta,\beta')
:=
\frac{Z_{L,t,\beta'}^2}
{\overline Z_{t,\beta}
\overline Z_{t,2\beta'-\beta}}.
\]

This is a deterministic certificate for the true population path distance.
It does not use empirical CESS.

### Fail-closed path rule

Starting from \(\beta=0\), select the largest registered numerical
\(\beta'>\beta\) for which

\[
\underline r_t(\beta,\beta')\ge r_0=0.8.
\]

If no positive increment is certified, or the bridge budget is exhausted
before \(\beta=1\), stop with NO-GO. A terminal step below the floor is
forbidden. The target remains ordinary Bayes because only the numerical path
depends on the observed current response.

### Frozen L=3 counter-certificate

Evaluating the exact 42-state size-three slice and the analytic tail envelope
on the four registered beta increments gives the following worst lower bounds:

| Fixture | Minimum certified \(r_{\rm ESS}\) | Worst bridge |
|---|---:|---|
| AC | 4.50007105e-8 | observation 8, 0.75 to 1 |
| AD | 2.18354984e-7 | observation 8, 0.75 to 1 |
| AE | 0.0096027394 | observation 8, 0.75 to 1 |

All are far below 0.8. Moreover, when \(\beta'=\beta\), the same interval
construction cannot exceed

\[
\left(Z_{L,t,\beta}/\overline Z_{t,\beta}\right)^2
=(1-\bar\tau_{L,t,\beta})^2.
\]

More strongly, log-convexity of the finite-slice normalizer gives

\[
Z_{L,t,\beta'}^2
\le Z_{L,t,\beta}Z_{L,t,2\beta'-\beta},
\]

so every candidate step satisfies

\[
\underline r_t(\beta,\beta')
\le
\left(
\frac{Z_{L,t,\beta}}{\overline Z_{t,\beta}}
\right)
\left(
\frac{Z_{L,t,2\beta'-\beta}}
{\overline Z_{t,2\beta'-\beta}}
\right)
\le
\frac{Z_{L,t,\beta}}{\overline Z_{t,\beta}}.
\]

The first beta-zero bridge state whose retained-evidence fraction falls below
0.8 is:

| Fixture | First blocked observation | \(Z_{L,t,0}/\overline Z_{t,0}\) |
|---|---:|---:|
| AC | 2 | 0.339598532 |
| AD | 2 | 0.324976838 |
| AE | 3 | 0.739431171 |

At each state, every possible positive next step therefore has a certified
relative-ESS lower bound below 0.8. A loose tail interval cannot be repaired by
taking a smaller bridge step. The current L=3 certificate cannot construct a
valid population-relative-ESS path and G4 is NO-GO.

The relevance of adjacent \(L_2\) distance and kernel mixing follows
[Marion, Mathews, and Schmidler, Finite Sample Bounds for Sequential Monte Carlo](https://arxiv.org/abs/1807.01346).

## 8. G5 proof and current NO-GO: decision error

### Terminal finite-computation bound

Require a terminal equal-weight resampling followed by \(m\) steps of the G3
prior-independence kernel at the final target. Let

\[
b_m=(1-\underline\epsilon_T)^m.
\]

For any \(0\le\varphi\le1\), every terminal particle marginal has bias at most
\(b_m\). Let each independent island return a bounded estimate
\(Y_k\in[0,1]\). Islands are independent even when particles inside an island
are dependent. Hoeffding's inequality gives, simultaneously for \(F\)
registered scalar functionals,

\[
\left|
\frac1K\sum_{k=1}^K Y_k-EY_k
\right|
\le
s_{K,F,\alpha}
:=
\sqrt{\frac{\log(2F/\alpha)}{2K}}
\]

with probability at least \(1-\alpha\).

For finite-slice validation,

\[
e_{\rm total}
\le
\bar\tau_L+b_m+s_{K,F,\alpha}+e_{\rm num}+e_{\rm boundary}.
\]

For a production open chain, the tail term is not part of the target error;
it remains a conservative exact-reference cross-check. Its particle law still
uses \(b_m+s_{K,F,\alpha}\).

This bound intentionally does not pretend that more correlated particles
inside one island remove finite-\(N\) bias.

### MAP-class decision

If every class probability has a simultaneous interval
\([L_c,U_c]\), release class \(\hat c\) only when

\[
L_{\hat c}>\max_{c\ne\hat c}U_c.
\]

Then \(\hat c\) is the exact MAP class for every posterior law in the
certificate set. Otherwise return uncertified/abstain.

The weaker norm statement remains

\[
\operatorname{Regret}_{0-1}\le2\delta_C
\]

when the class-pushforward total variation error is at most \(\delta_C\).

### Quantized class-EIG continuity

Let \(Q(Y)\) use the seven frozen thresholds and have \(R=8\) bins. If the true
and computed joint laws of \((C_\star,Q(Y))\) differ in total variation by
\(\delta\), define

\[
\omega_1(\delta)=0,\qquad
\omega_m(\delta)=
\min\{\log m,\,
h_2(\delta)+\delta\log(m-1)\},\quad m\ge2,
\]

using the sharp second expression when
\(\delta\le1-1/m\). The entropy-continuity inequality gives

\[
|I_P(C_\star;Q)-I_{\widehat P}(C_\star;Q)|
\le
\omega_C(\delta)+
\omega_R(\delta)+
\omega_{CR}(\delta),
\]

where \(C\le K_B^{7I}\). This envelope is added outward to the existing
data-processing lower bound and maximum-entropy upper bound.

For action utilities with simultaneous intervals
\([\underline U_a,\overline U_a]\), targeted acquisition is permitted only if

\[
\underline U_{\hat a}>
\sum_a p_{\rm ref}(a)\overline U_a.
\]

Otherwise the registered reference action is used. Conditional on valid
intervals, this proves that the selected policy is not worse than the
reference policy under the registered model-relative utility.

Entropy-continuity reference:
[Audenaert, A sharp continuity estimate for the von Neumann entropy](https://arxiv.org/abs/quant-ph/0610146).

### Current counter-certificate

For the current finite-slice resident family, the omitted-posterior term alone
is at least 0.923809 on all three VR.8 fixtures and exceeds 0.999 on two. It
makes every nontrivial full-posterior MAP, total-variation, or quantized-EIG
interval vacuous. In a future production open chain the tail is not an
additive target error, but this finite-slice exact-reference comparison remains
vacuous. Independently, the open-chain terminal mixing bound needs 871 to
176,972 prior-independence MH steps per particle merely to drive its bias term
below 0.01, while the current resident schedule supplies one terminal move.

No allocation among island randomness or numerical error can repair these two
dominant terms at the frozen budget. G5 is therefore NO-GO.

## 9. G6 proof: response and role noninterference

The following access table is normative.

| Object | Initial \(X_0\) | Initial \(Y_0\) | Visible candidate \(X\) | Hidden candidate \(Y\) | Mechanism-development outcomes | Real validation \(Y\) | Held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grammar and target priors | no | no | no | no | no | no | no |
| Target likelihood update | yes | yes | no future row | no | no | no | no |
| \(C_\star\) and response scale | yes | yes | yes | no | no | no | no |
| Base proposal \(q_0\) | no | no | no | no | no | no | no |
| Bridge path | current row only | current row only | current action only | only after that response becomes observed history | no | no | no |
| Tail and mixing certificates | yes | yes | visible action domain | no | no | no | no |
| Error thresholds and decision rules | no | no | action identifiers/distribution only | no | no | no | no |
| Evaluation/audit | response-free synthetic fixtures only | response-free synthetic fixtures only | no real pool | no | may report but never alter contract | no | no |

### Proposition G6.1: target noninterference

The grammar, size prior, discrepancy spike/kernel probabilities, NIG
hyperparameters, operator language, and measurement-error state are fixed
without response input. The likelihood uses only observations already in the
scientific history.

### Proposition G6.2: estimand freezing

\(C_\star\) is computed from the declared \(H_0\) and selection-visible action
domain, hashed, and then frozen. Later posterior mass is pushed through that
same map. Future responses cannot relabel a state.

### Proposition G6.3: proposal noninterference

The admitted proposal is exactly the response-independent prior. No learned
proposal enters this version. Response-dependent likelihood values occur only
inside the exact MH acceptance ratio and do not change proposal support or the
target.

### Proposition G6.4: path noninterference

The path may use the currently observed row to choose a numerical beta
sequence, but every intermediate density is a fractional bridge to the same
ordinary Bayesian endpoint. No future response or validation role enters.

### Proposition G6.5: decision noninterference

All formulas, probability levels, budget \(B\), class bins, confidence level,
and fallback rule are frozen before any mechanism-development result. Observed
development failures can only produce a disposition, never a threshold
change.

The mathematical role proof is complete. A later implementation would still
need role-typed APIs and capability-denial tests before it could claim the
same property.

## 10. Overall theorem and consequence

### Theorem

Under Amendments A1--A4 and the registered positive hyperparameters:

1. the grid-restricted operational estimand is a valid finite measurable
   pushforward on the complete support;
2. the countably open posterior is proper;
3. its finite-slice evidence and posterior-tail upper bounds are explicit;
4. the prior-independence MH kernel is target-correct on the complete support;
5. population relative ESS and terminal decision errors have explicit
   fail-closed certificates; and
6. the scientific data roles can be separated without future-response access.

For the current \(L=3\), 32-bridge, one-terminal-move resident family, the tail,
path, and mixing bounds are too weak by orders of magnitude. Therefore no
decision-level finite-sample fidelity certificate exists at the frozen budget.

### Required next research, not implementation

The proof package sends the project further upstream. Before any mechanism
code, a new target-level contract must solve both:

1. a computationally sharp posterior-tail certificate that does not enumerate
   billions of raw syntactic aliases; and
2. a target-correct proposal with a practically meaningful minorization or
   finite-sample mixing bound.

A canonical semantic prior or another representation-controlled target may be
needed. Such a change is a new posterior target version, not a proposal tweak
and not VR.9.

Until those two mathematical objects are proved, the status remains:

- G4: NO-GO;
- G5: NO-GO;
- inference implementation: blocked;
- confirmatory freeze: blocked;
- predictive calibration: blocked;
- real data, acquisition, and held-out: blocked.
