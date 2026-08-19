# P3F.4-CERT.1 semantic-envelope certification layer

Status: **DEVELOPMENT PASS; UNSEEN CONFIRMATORY CERTIFICATION FREEZE ELIGIBLE**

Resident-SMC integration status: **BLOCKED**. This package implements and
tests a static certification layer. It does not change, invoke, or authorize a
new resident-SMC mechanism.

## 1. Claim and target boundary

The frozen raw grammar target remains

\[
p(T)=p(S=s)p(T\mid S=s),\qquad
p(S=s)=(1-\rho)\rho^{s-1},\qquad
p(T\mid S=s)=N_s^{-1},
\]

where \(N_s\) is the number of raw well-typed ASTs with \(s\) nodes. The
certification layer does not replace this prior with a uniform prior over
semantic classes.

Production scientific functionals must be constant on exact polynomial
equivalence classes. Raw-AST identity may remain a diagnostic, but it is not a
production estimand under the quotient implementation. If a future claim
requires raw-AST posterior probabilities, a representative must be sampled
conditionally with its exact within-class raw multiplicity; that operation is
not authorized here.

The current observed history may be used to evaluate ordinary or fractional
target likelihoods and to construct a target-correct proposal frozen within a
bridge. Future responses, held-out state, real validation outcomes, acquisition
responses, and discovery outputs remain inaccessible.

## 2. Exact semantic multiplicity theorem

Let \(\kappa(T)\) be the exact integer-coefficient polynomial key of raw AST
\(T\). Define

\[
C_s(k)=\#\{T: |T|=s,\ \kappa(T)=k\}.
\]

For the registered terminals \(1,x_1,\ldots,x_d\), unary negation, and binary
addition and multiplication, the dynamic program is

\[
\begin{aligned}
C_1(k)
&=\mathbf 1\{k=1\}+\sum_{j=1}^d\mathbf 1\{k=x_j\},\\
C_s(k)
&=C_{s-1}(-k)\\
&\quad+\sum_{\ell=1}^{s-2}\sum_{a+b=k}
C_\ell(a)C_{s-1-\ell}(b)\\
&\quad+\sum_{\ell=1}^{s-2}\sum_{ab=k}
C_\ell(a)C_{s-1-\ell}(b).
\end{aligned}
\]

The raw shell count obeys

\[
N_1=d+1,\qquad
N_s=N_{s-1}+2\sum_{\ell=1}^{s-2}N_\ell N_{s-1-\ell}.
\]

Induction on \(s\), summing the semantic recurrence over \(k\), gives

\[
\sum_k C_s(k)=N_s.
\]

Therefore the exact quotient prior through cutoff \(J\) is

\[
w_J(k)=
\sum_{s=1}^J
(1-\rho)\rho^{s-1}\frac{C_s(k)}{N_s},
\]

and it conserves the raw prior mass:

\[
\sum_k w_J(k)=1-\rho^J.
\]

This is a regrouping of the original raw target, not a new semantic prior.

### Registered one-dimensional count reduction

At \(J=17\):

| Quantity | Exact value |
|---|---:|
| Cumulative raw ASTs | 5,924,484,194 |
| Size--polynomial cells | 31,209 |
| Distinct polynomial keys | 13,574 |
| Quotient prior mass | 0.9999998282013081 |
| Absolute mass error in the development run | 1.11e-16 |

The likelihood and structure-wise projected discrepancy basis depend on the
polynomial evaluation vector and its span, not on the raw serialization.
Consequently every registered component likelihood is constant within one
exact polynomial class, and one likelihood evaluation per class and
discrepancy state is sufficient.

## 3. Deterministic posterior-tail certificate

Let \(m_\lambda(z)\) be the collapsed marginal likelihood of component \(z\)
under non-negative likelihood powers \(\lambda=(\lambda_1,\ldots,\lambda_n)\),
and let

\[
\nu=\sum_i\lambda_i.
\]

For the registered Gaussian likelihood and Normal--Inverse-Gamma prior,
positive definiteness gives the component-uniform bound

\[
m_\lambda(z)\le M_\nu,
\]

where

\[
M_\nu=
(2\pi)^{-\nu/2}
\frac{\Gamma(a_0+\nu/2)}{\Gamma(a_0)}
b_0^{-\nu/2}.
\]

The determinant ratio is at most one and the posterior noise scale is at least
\(b_0\), so the bound holds for spike and every registered discrepancy state,
including fractional powers above one used by the second-moment calculation.

Let

\[
Z_{J,\lambda}
=\sum_k w_J(k)\sum_d p(d)m_\lambda(k,d)
\]

be the exact semantic-core evidence. The unresolved raw tail satisfies

\[
0\le Z_{>J,\lambda}\le U_{J,\lambda}:=\rho^J M_\nu.
\]

Hence

\[
Z_{J,\lambda}\le Z_{\infty,\lambda}
\le \overline Z_{J,\lambda}
:=Z_{J,\lambda}+U_{J,\lambda},
\]

and

\[
\Pr(S>J\mid H,\lambda)
\le
\overline\tau_{J,\lambda}
:=
\frac{U_{J,\lambda}}
{Z_{J,\lambda}+U_{J,\lambda}}.
\]

### Final-history development results

| Fixture | Exact core evidence | Tail evidence upper | Posterior-tail upper |
|---|---:|---:|---:|
| AC | 0.4091589627123654 | 0.0009688196347819 | 0.0023622384936650 |
| AD | 0.4346689175633862 | 0.0009688196347819 | 0.0022239111813704 |
| AE | 30.80299029407879 | 0.0009688196347819 | 0.0000314511401345 |

All three are below the registered 0.01 tail ceiling. The previous \(L=3\)
certificate failed because it assigned the uniform likelihood envelope to
6.4% raw prior mass. The new certificate evaluates sizes 1--17 exactly after
semantic aggregation and assigns the envelope only to \(\rho^{17}\).

## 4. Certified bridge-path theorem

For one bridge \(\beta<\beta'\), population relative ESS is

\[
r_{\mathrm{ESS}}(\beta,\beta')
=
\frac{Z_{\beta'}^2}
{Z_\beta Z_{2\beta'-\beta}}.
\]

The semantic-core intervals give the deterministic lower certificate

\[
\underline r_{J}(\beta,\beta')
=
\frac{Z_{J,\beta'}^2}
{\overline Z_{J,\beta}
 \overline Z_{J,2\beta'-\beta}}
\le r_{\mathrm{ESS}}(\beta,\beta').
\]

At every observation the registered algorithm evaluates the grid

\[
\{0,1/32,2/32,\ldots,1\}
\]

and chooses the largest \(\beta'>\beta\) for which

\[
\underline r_J(\beta,\beta')\ge0.8.
\]

If no positive step exists or 64 steps are exceeded, the certificate returns
NO-GO. A forced terminal step is forbidden.

The old fixed quarter grid is not retained: at the first observation its first
step had certified lower relative ESS 0.339 for AC, 0.317 for AD, and 0.793 for
AE. The new rule is therefore a mathematical path correction, not a cosmetic
relabeling of the old schedule.

### Full eight-observation development results

| Fixture | Total certified bridges | Worst lower relative ESS | First-observation path |
|---|---:|---:|---|
| AC | 16 | 0.8042550126 | 0, 1/32, 3/32, 6/32, 11/32, 19/32, 1 |
| AD | 16 | 0.8042341050 | 0, 1/32, 3/32, 6/32, 11/32, 19/32, 1 |
| AE | 12 | 0.8013162362 | 0, 7/32, 21/32, 1 |

Every observation path reached one below the 64-step fail-closed budget.

## 5. Envelope independence proposal and mixing theorem

Use a hybrid state space:

- the exact semantic quotient for raw sizes \(s\le J\); and
- the original raw AST/component state for \(s>J\).

Let the unnormalized target be

\[
\gamma(z)=p(z)m_\lambda(z),
\]

where \(p(z)\) denotes the quotient prior mass in the core and the original raw
prior probability in the tail. Define

\[
C_{J,\lambda}=Z_{J,\lambda}+U_{J,\lambda}
\]

and the independent proposal

\[
q_{J,\lambda}(z)=
\begin{cases}
\gamma(z)/C_{J,\lambda}, & z\text{ in the semantic core},\\
p(z)M_\nu/C_{J,\lambda}, & z\text{ in the raw tail}.
\end{cases}
\]

This proposal is exactly normalized. It is sampled by a categorical draw over
the finite semantic core versus the analytic tail, followed in the tail by the
memoryless geometric size law and the existing exact raw-AST prior sampler.

For the normalized target \(\pi=\gamma/Z\),

\[
\frac{q(z)}{\pi(z)}
=\frac{Z}{C_{J,\lambda}}
\begin{cases}
1, & z\text{ in the core},\\
M_\nu/m_\lambda(z), & z\text{ in the tail},
\end{cases}
\]

so

\[
q(z)\ge \epsilon_{J,\lambda}\pi(z),
\qquad
\epsilon_{J,\lambda}
\ge
\underline\epsilon_{J,\lambda}
:=
\frac{Z_{J,\lambda}}
{Z_{J,\lambda}+U_{J,\lambda}}.
\]

The independence-MH kernel is therefore uniformly ergodic and

\[
\sup_x\|P^m(x,\cdot)-\pi\|_{\mathrm{TV}}
\le(1-\underline\epsilon_{J,\lambda})^m.
\]

This is the standard independent-Hastings domination condition of
[Mengersen and Tweedie (1996)](https://doi.org/10.1214/aos/1033066201).

The MH correction is explicit:

- core to core: acceptance one;
- tail to core: acceptance one;
- core to tail: \(\min(1,m(z')/M_\nu)\);
- tail to tail: \(\min(1,m(z')/m(z))\).

At the final development histories:

| Fixture | Minorization lower | One-step TV upper | Steps for TV at most 0.01 |
|---|---:|---:|---:|
| AC | 0.9976377615063350 | 0.0023622384936650 | 1 |
| AD | 0.9977760888186296 | 0.0022239111813704 | 1 |
| AE | 0.9999685488598655 | 0.0000314511401345 | 1 |

The same normalizer intervals also supply the adjacent-distribution control
used by finite-sample SMC analysis; see
[Marion, Mathews, and Schmidler](https://arxiv.org/abs/1807.01346).

## 6. Access and adaptation contract

| Operation | Allowed information | Forbidden information |
|---|---|---|
| Semantic multiplicity DP | Grammar, feature count, \(\rho\), cutoff | All responses |
| Semantic design catalog | Selection-visible action grid | Future or held-out actions if not selection-visible |
| Core likelihood/evidence | Current target history and current bridge power | Future, validation, held-out, acquisition, real-efficacy responses |
| Bridge selection | Frozen grid, current normalizer certificates | Empirical particle ESS used as a substitute for the analytic lower bound |
| Proposal construction | Current target history, frozen within the bridge, exact \(q\) | Unrecorded learned/local proposal probabilities |
| Gate decision | Registered thresholds only | Threshold changes after observing results |

The cutoff may increase as a fail-closed numerical certification rule, but it
does not change the posterior target or scientific estimand. A confirmatory
experiment must freeze the cutoff-growth rule, candidate grid, thresholds,
and maximum budget before new responses are evaluated.

## 7. Development disposition

The static development run passed all five registered decisions for AC, AD,
and AE:

- exact semantic prior-mass conservation;
- component likelihood envelope;
- posterior-tail ceiling;
- certified bridge path; and
- envelope-proposal mixing bound.

This closes the three upstream mathematical defects identified in the G0--G6
review for the registered one-dimensional grammar. It does **not** yet close
the full paper gate:

1. the envelope proposal has not been integrated into resident-SMC;
2. no unseen confirmatory certification fixture has been opened;
3. finite-particle/island error in G5 still needs to be recomposed with the new
   tail, path, and mixing constants; and
4. real data, predictive calibration, acquisition, held-out, efficacy, and
   discovery remain blocked.

The only authorized next step is an unseen confirmatory certification freeze,
followed by a separate reviewed resident-SMC integration if that freeze passes.
