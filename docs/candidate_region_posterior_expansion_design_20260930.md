# Candidate / region posterior expansion: architecture and correctness boundary

Status: **response-free research primitive** on top of `main` at `92241b7`.
No formal experiment, measured-pool policy, new efficacy claim, or held-out
access follows from this change. The prior negative results stay immutable.

## What the current system actually computes

| Stage | Current implementation | Missing distinction |
| --- | --- | --- |
| Proposal | `agent.py`, `scientific_runtime.py`, `system_ablation.py`: core, optional engines, and typed LLM synthesis; source-first projections support matched ablations | In the recent external typed-synthesis profile the inner free-form LLM equation proposer was disabled. The Scientist called the provider, but no synthesis row reached final top-k; do not treat Full == No-LLM as evidence against an unexecuted proposer. |
| Individual admission | `source_stacking.py:filter_fold_safe_source_candidates`: each optional support versus the complete core on two arbitration folds | Global log density gain can hide a local improvement; strict positive gain on **both** folds is a selection criterion on those same data, not a guarantee of population safety. |
| Conditional complementarity | `filter_conditionally_complementary_engine_candidates`: engine candidate versus fitted core-plus-LLM stack | Reference and expanded weights are optimized on the same arbitration rows used to declare fold gains; competition between candidates is not assessed jointly. |
| Bank / prior | `bank_selection.py`: enumerate protected-core portfolios, fit source weights, rank by their own class entropy or certified conditional class-risk utility. `pcpi_adapter.py` divides each source's prior weight equally between its retained supports | Candidate-specific and regional posterior contribution is unobserved; portfolios induce different class partitions and initial-risk scales. |
| Source decision audit | `marginal_influence.py`: leave one source out; score its chosen action in the Full bank's class-EIG target | A source-level, initial-step, model-relative regret check; not the candidate's contribution to the same external decision loss across regions or across steps. |
| Acquisition / reporting | `realized_drr.py`: update a frozen finite-bank posterior; compare independent predictive MSE and log score if supplied. `robust_common_target.py`: finite-law interval selection primitive | The shared external report is optional and not wired to the frozen runner. The robust selector has no calibrated law set or runner integration. Numerical separation alone does not imply real-law coverage. |

This is why structure novelty, internal utility, and Full > No-LLM may diverge.
In particular, a false candidate can raise its own bank's discrimination value;
another candidate can help only in a small covariate region and fail a global
admission rule. These are mechanisms, not diagnoses established for every task.

## New exact finite reference module

`hypothesis_mvp/discovery/candidate_region_expansion.py` separates the three
arrows: candidate -> local posterior mass -> common-target decision effect.
`FrozenAxisRegions` assigns regions using only frozen covariates. It is a simple
reference partition, not an adaptive learned region tree. The prior for each
region is explicitly supplied; the caller must freeze its source/candidate
mass before opening selector observations. The first expert is the protected
core predictive distribution, followed by distinct candidate experts.

Conditional on **fixed**, normalized expert predictive densities
`f_k(y | x, H0)`, the selector model is

```
Z_r ~ Categorical(pi[r, :])
Y_i | Z_{r(x_i)}=k, x_i, H0 ~ f_k(. | x_i, H0)
p(Z_r=k | D) = pi[r,k] * exp(sum_{i:r(x_i)=r} log f_k(y_i|x_i,H0)) / normalizer
```

The implemented likelihood accumulator is exact for this conditional model.
It is **not** the original PCPI posterior over symbolic structures and
coefficients. If an expert is refit after every response, simply appending
its re-fitted predictive score double-counts data and invalidates this
identity. Either keep the expert densities fixed over a registered segment or
derive a coherent sequential marginal-likelihood update for a new protocol.

For candidate `h`, `log BF[h,core,r]` is the difference of cumulative local
log likelihoods; `p(Z_r=h|D)` reports local posterior mass. Candidate removal
is defined **before** looking at its selector evidence: its declared prior mass
goes to the core and the complete remaining likelihood is re-normalized.
This avoids interpreting a raw posterior renormalization as the result of a
different prospective bank.

The target is a common covariate grid with weights `nu_i`, a fixed squared
loss, and an explicitly supplied plausible law `q`. For the predictive-mean
decision, the loss is `sum_i nu_i (sum_k p(Z_{r_i}=k|D) mu_ik - m_q(x_i))^2`.
`candidate_region_contribution` gives the signed reduction in that loss versus
the pre-evidence ablation. `action_values_under_law` integrates this same loss
after one action response over a finite response grid supplied by `q`, using
the Bayesian selector update. It can return a **negative** value when an
apparently novel but wrong candidate is reinforced. No outcome or report-set
response is used to select an action in this module; a law/target mean supplied
from actual report outcomes may only be used in an offline report after the
selection ledger is closed. The finite grid does not certify continuous
quadrature error.

`candidate_action_contribution` connects the three arrows directly: it takes
the candidate's regional posterior mass and Bayes factors, selects the best
worst-law action with and without that candidate, then reports the signed
difference in after-action loss under each **same** specified law and target.
Its action-regret field evaluates the ablated policy's action in the Full
selector, still under the common plausible laws. Identity mismatch fails.
It does not promote a model-relative value to a real-response guarantee.

## Research integration gates, in order

1. **Freeze source and region priors.** The existing source gate may provide a
   development-only source allocation; split each source's mass across its
   candidate experts by a predeclared hierarchical prior. Keep a protected
   core reserve. Freeze the region map from registered covariates without
   responses. Do not estimate local priors from the rows that test them.
2. **Freeze experts and distinguish data roles.** Fit symbolic coefficients on
   H0, admit candidates on a separate arbitration subset, fit the selector on
   an additional strict-prefix subset, and reserve a distinct report set.
   Audit row identities, provenance, and every fitting/selection dependency.
   Existing benchmark discovery-validation rows are already consumed.
3. **Derive the production predictive density.** The current finite PCPI
   `ReferenceBank` has one global structure per posterior member. A regional
   selector is a separate modular predictive model. Do not multiply its local
   masses into the old class posterior or label the result the same scientific
   target. An integrated joint generative model would require an explicit
   prior over regional structures and coherent likelihood/update proof.
4. **Define plausible laws and common action loss.** For each action, freeze
   response laws from independent calibration, including a discrepancy law;
   check conditional/shift coverage by region and account for selecting many
   candidates/regions. Compute robust lower/upper *loss reduction* for the
   same target; use `select_maximin_common_target` only with compatible law
   identities and certified numerical intervals. If coverage or rank is
   unresolved, use the frozen reference action. Avoid tuning a confidence
   cutoff on prior negative tasks.
5. **Prove lifecycle composition before measured runs.** New protocol identity,
   source integrity, candidate-pool hash, strict decision-before-reveal,
   exactly-once updates, checkpoint replay and numerical error bounds must pass
   response-free exact references. Then freeze a user-executed development
   protocol with matched Full/No-LLM/random and a disjoint report target.

## What should change in the design, beyond this reference

* Use partial pooling across neighboring regions so a low-count region does
  not appear certain from one observation. A hierarchical prior or smooth
  covariate-dependent stacking is plausible, but stacking weights optimized
  for log score are **not** automatically posterior probabilities of scientific
  laws. State which estimand each weight represents.
* Make candidate deletion a leave-one-candidate-out and leave-one-region-out
  **common-law loss** audit. Retain source-level checks for provenance, but do
  not require every candidate to change the top action: a candidate can
  improve prediction while the action stays the same.
* Charge structural search for the number of attempted candidates and regions.
  A repeated fold check on the same selected portfolio is optimistic; use a
  disjoint audit or a simultaneous selection-aware uncertainty bound.
* Compare action value and external predictive benefit separately. The former
  depends on calibrated plausible laws; the latter is computed only after
  revelation on the untouched report set. Preserve negative and abstention
  counts at the independent-task level.

The controlled fixture tests show exact local Bayes accumulation, unchanged
unobserved regions, prospective core-mass ablation, a helpful regional
candidate, a candidate-specific action change on a common law, a false
candidate with negative external action value, and fail-closed shape/identity
checks. They make no efficacy claim.

Related primary research: Yao et al., *Bayesian hierarchical stacking: Some
models are (somewhere) useful* (2021), arXiv:2101.08954; Yao et al., *Using
stacking to average Bayesian predictive distributions* (2018),
arXiv:1704.02030; Tang et al., *Representative, Informative, and
De-Amplifying* (AISTATS 2026), PMLR 300.
