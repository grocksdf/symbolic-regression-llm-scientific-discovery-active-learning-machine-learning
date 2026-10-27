# Global Protected Admission contract

This is a new method identity. It does not modify or reinterpret historical
regional-admission results.

## Target alignment

A candidate admitted by this rule becomes a global hypothesis-bank member.
Therefore evidence from one favorable region cannot by itself grant global
epistemic authority.

For candidate \(k\), independent admission observations \(i\), fixed core
predictive density \(p_0\), fixed candidate predictive density \(p_k\), and
registered regions \(r\), define

\[
L_k=\sum_i \log\frac{p_k(y_i\mid x_i)}{p_0(y_i\mid x_i)},\qquad
L_{k,r}=\sum_{i:R(x_i)=r}\log\frac{p_k(y_i\mid x_i)}{p_0(y_i\mid x_i)}.
\]

With \(K\) attempted candidates and frozen familywise level \(\alpha\), a
candidate is admitted only when

\[
L_k>\log(K/\alpha)
\quad\text{and}\quad
\min_{r\in\mathcal R} L_{k,r}\ge 0.
\]

The first condition is the multiplicity-adjusted fixed-law evidence gate. The
second is a zero-tolerance direction constraint: a global law cannot be
authorized by compensating a negatively supported registered region with a
different favorable region. Empty or nonfinite regions fail closed.

## Boundaries

- Gap-audit responses do not enter this decision.
- Admission responses are disjoint from fit and Gap-audit roles.
- Thresholds are fixed before a new experiment.
- No historical task, formula, direction, score, or result appears in the rule.
- Regional admission remains available only under its historical method
  identity.
- This correctness contract does not establish efficacy or authorize a real
  experiment.
