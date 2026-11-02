# CB-SI/FCF v3 — Feasibility-Conditioned Counterfactual Frontier

Status: **implementation + response-free correctness tests**. This is an
experimental development mechanism, not a claim of algorithmic priority or
independent scientific discovery. Version v3 is incompatible with the frozen
v1/v2 edit semantics and results. No old confirmation can be reused.

## Hypothesis and explicit boundaries

CB-SI originally updates a standard Normal-Gamma operator bandit from
matched Bank-conditional improvement. The earlier implementation loses
training opportunities to inapplicable sites, missing donor material,
algebraically identical proposals and controls whose predictive supports
are indistinguishable. Its candidate archive is static.

FCF tests two additional mechanisms under a new, separately staged protocol:

1. **Feasibility-conditioned contextual operator selection** based only on
   syntax, construction grammar and already-visible search-side feedback.
2. **Bounded multi-generation replay:** a positive search-gain child may become
   the parent of a future edit; the original predictive Bank remains fixed
   while candidate Bank-conditional improvement is scored.

All labels used to update the policy come from two designated cross-fitting
search roles, not the one-shot independent development evaluation role.
No reporting/test/OOD/untouched-confirmation labels or covariates enter the
search. Parent/donor heuristic ranking may use *development* labels.

## Formal objects and scoring (not a scientific posterior)

At edit step `t`, the exogenous opportunity is `(f_t, d_t, s_t, seed_t)`.
The new context `c_t` is `root/internal × {binary, unary, function, leaf}`.

A response-free preflight returns only operators with grammar-valid candidate
support different from the parent and already-seen supports. No preflight
uses responses, and no additional primary slot is minted if preflight fails.

For operator `o` and context `c`, model:

- `P_novel(o,c)`: Beta(1,1) prior, updated once for every nominal primary
  attempt; success means *valid and novel and scored*.
- `P_pair(o,c)`: Beta(1,1) prior, updated for pair identification; a missing
  control is a failure *indicator*, **never a numerical credit**.
- `G_raw(o,c)`: Normal-Gamma working reward model, updated only for scored
  valid/novel primaries.
- `G_cf(o,c)`: a different Normal-Gamma working reward model, updated only
  for matched primary/control requests that both produce distinct valid and
  novel Bank supports with finite paired gains.

The policy samples a heuristic score:

    score(o,c) = draw(P_novel(o,c)) * (
        draw(G_raw(o,c)) + 0.5 * draw(P_pair(o,c)) * draw(G_cf(o,c)))

The two reward components use the corresponding site-specific model after at
least two site-specific observations and otherwise use operator-level reward
observations as a **backoff heuristic**. It is not a coherent hierarchical
posterior or a causal posterior of scientific truth. The score optimizes
Bank-conditional *predictive development gain* while preserving evidence of
matched structural credit; the coefficient `0.5` is frozen in the protocol,
not tuned against evaluation outcomes.

Matched credit, for the registered two search folds, is:

    (G_primary - G_parent) - (G_control - G_parent)
        - lambda_complexity * (delta_complexity_primary
                              - delta_complexity_control)

Only a validated pair can call `counterfactual_edit_credit`.
Counterfactual missingness and observability are separately reported. A
positive observational credit under adaptive search is not a proof of a
causal operator effect or Bayesian calibration.

## Edit and novelty changes

- `subtree_composition` means **multiplication**, while `correction_term`
  means **addition**. This removes a redundant action from the new protocol.
- Operator substitutions have syntax-specific eligible sites, and donor
  operators require donor material.
- `factor_or_expand` and `structural_simplification` remain available in the
  structural engine as representation edits; they are not eligible as
  independent *semantic invention* actions in FCF.
- Finite-domain approximate numerical agreement is a separate diagnostic,
  never proof of symbolic equivalence. Small bounded expressions receive
  a SymPy identity check. Structural, canonical and fixed-formula support
  identities are kept distinct. Empty/nonfinite covariate domains fail closed.
- Preflight checks are additional *planning computation*, not free compute.
  The result explicitly reports preflight checks, predictive scorer calls
  and wall-time; equal nominal edit slots do not establish equal CPU costs.

## Trajectory and equal-budget ablations

| Arm | Initial opportunities | Policy | Descendant reuse |
|---|---|---|---|
| `cbsi_full` | fixed | standard matched-credit Thompson | no |
| `cbsi_feasible` | same fixed opportunities | contextual feasibility/raw+credit | no |
| `cbsi_frontier` | same initial pool | same contextual policy | yes; adaptive on odd blocks |
| `bank_greedy`, `standard_thompson`, `blind_random` | same fixed opportunities | legacy | no |
| fitness/residual/influence proxies | differing proxy opportunities | legacy | no |

There are **nine** arms. Each receives the same pre-registered number of
primary/control *reserved slots*, not necessarily the same number of valid
or scored candidates, and not necessarily the same measured CPU budget.
The frontier cap is 8 formulas and maximum generation depth 3. Only a novel
candidate with positive cross-fit Bank gain enters the frontier. Odd blocks
may edit the best available search-side descendant; even blocks use the
original exogenous opportunity. This rule is predeclared and deterministic.
Adaptive descendants **are not opportunity-matched** to static arms after
initialization; only budgets and initial data are matched.

## Development metrics and gate

For each registered task, score exactly one selected candidate per arm on
`gap_admission` *after* search is frozen. Report all failures and every arm,
including negative task gains. Summarize within-task seeds before resampling
tasks. The pre-specified two-sided paired task bootstrap must have a strictly
positive lower 95% limit for all three comparisons:

1. `cbsi_feasible - cbsi_full`
2. `cbsi_frontier - cbsi_feasible`
3. `cbsi_frontier - bank_greedy`

The frontier mean must also exceed zero and the strongest legacy baseline.
This is a **development screening rule only**, not the project's sealed
confirmation or research-paper efficacy gate. Selecting the best baseline
on development is descriptive, not a registered independent significance
test. Changing a cost weight, contextual weight, operator set, task, seed,
budget, or screening criterion requires a new protocol identity.

## Freeze and provenance

`prepare_cbsi_v3_freeze.py` accepts an existing complete v1 or v2 CB-SI
freeze and creates a new v3 **draft** with `execution_authorized=false`.
It preserves data/split/candidate artifacts and their hashes, refreshes only
the two changed code identities, records new Git commit/tree/clean-state
identities, and requires a clean committed worktree. The requested new
budget (if any) is recorded as a deliberate new protocol choice for all
arms. The runner requires fresh authorization and identical source identities.
`finalize_cbsi_v3_evidence.py` independently checks result completeness,
reserved-slot accounting, null credit on invalid pairs, request lineage,
run-freeze identity, and zero Provider/PySR calls; it never reads HDF5.

## Remaining limitations

No real benchmark efficacy test, full dependency regression, sealed
confirmation, paper novelty search or literature priority claim is included
in this code-only patch. The cross-fitting search roles are reused over
many adaptive decisions, so the search gains are optimistically selected;
only the untouched evaluation role at candidate-selection time can assess
that overfitting, and even it is development evidence. If the upstream
frozen candidate archive already used a role for candidate generation,
independence cannot be inferred from the role names alone. Verify its
lineage before using results as independent evidence.
