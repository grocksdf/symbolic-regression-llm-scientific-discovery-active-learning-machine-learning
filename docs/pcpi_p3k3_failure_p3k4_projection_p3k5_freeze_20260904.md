# P3K.3 terminal audit, P3K.4 projection, and P3K.5 freeze

## Preserved P3K.3 result

The source-frozen P3K.3 run stopped after 34,371.8 seconds at CCPP seed
2026080707, PCPI query 27. Queries 1--26 have complete decisions, matching
reveal receipts, and state-linked ledgers. Query 27 opened no response and
published a terminal failure because its representative safe set was empty.
Held-out remained closed and selection used no validation responses. The output
is terminal and must not be resumed, deleted, or relabelled as efficacy evidence.

## Root cause

The historical covariate guard admitted candidate (a) only when

\[
  D(S\cup\{a\},T)^2 \le D(S,T)^2,
\]

where (D) is the registered biased RBF-MMD, (S) is the opened design, and (T)
is the registered action domain. A finite candidate pool does not guarantee
that this set remains nonempty after every sequential removal. Terminal failure
was therefore a structural feasibility defect, not a floating-point tolerance
problem and not evidence about the repaired P3K probability model.

## P3K.4 minimum-violation projection

For every candidate define

\[
  \Delta(a)=D(S\cup\{a\},T)^2-D(S,T)^2,
  \qquad \Delta_*=\min_a\Delta(a).
\]

P3K.4 freezes the admissible set

\[
  A=\{a:\Delta(a)\le \max(0,\Delta_*)+\tau\},
\]

where (\tau) is the existing scale-aware roundoff allowance. If any
nonincreasing action exists, (\Delta_*\le0) and this is exactly the historical
safe set. Otherwise it contains only actions attaining the minimum possible
MMD increase. It is nonempty for every finite nonempty candidate pool and does
not switch acquisition utility: P3K EIG still ranks within (A).

The rule uses only opened covariates, candidate covariates, and the registered
action domain. It has no response, validation, held-out, dataset-name, seed, or
effect-size input and no fitted penalty. The no-data P3K.4 Gate verifies exact
reduction in the feasible case, minimum-violation optimality in the infeasible
case, nonemptiness, new method identity, and absence of experiment surfaces.

## P3K.5 formal boundary

P3K.5 changes only the representative feasibility rule and stage/source/config
identity relative to P3K.3. P3K shared-innovation inference, four likelihood
powers, matched baselines, datasets, seeds, splits, budgets, quadrature schedule,
runtime, assessment, and closed held-out boundary remain frozen. P3K.3's entry
point refuses current-source execution. P3K.5 remains failure-informed
real-development evidence and is not independent confirmation.
