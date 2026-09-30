# Quality-first proposal augmentation: development audit

This record concerns the supplied `dev_handoff_20260930.zip` and the immutable
2026-09-28 bounded realized pilot. It does not reclassify the failed pilot or
authorize a new real-data run.

## Evidence reconciliation

- All 102 files named in the package manifest match its SHA-256 values. All
  16 candidate artifacts also match the pilot freeze's `candidate_artifacts`.
- The three files under `04_source/` do **not** match the `files` SHA-256
  identities in the bounded realized pilot freeze. They describe later code.
  The original mainline source at commit `b82c592d1c8c8098a795b70601cd8f16177c1745`
  has the freeze-registered `realized_drr.py` hash
  `608593d00c7e849367ec6099a78c3b0ff30e2b81e05a6870f8daaea9364f149e`.
- That frozen implementation chooses `argmax(lower)` for the targeted policy,
  even when every lower bound is zero. It does not execute the later
  matched-random fallback on these zero-value actions. The 2026-09-29 repair
  does not repair or relabel the 2026-09-28 result.
- The `AISTATS_LLM_ADMISSION_AUDIT.json` in `03_context` covers six resolved
  Full artifacts from a different experiment (for example BPG7/seed91),
  whereas the realized pilot uses BPG15, CRK1, I.34.14_2_0 and MatSci1,
  seeds 71/72. Its `synthesis_final_topk_rows_total=0` is therefore **not**
  a statistic for the realized pilot.
- In the realized pilot's eight Full candidate artifacts, six have a typed
  LLM candidate in the DRR adapter's source-balanced input pool. This does
  not prove it survived the later capacity-four selector. The pilot only
  publishes selected target hashes, not the selected support list. Different
  target hashes alone cannot identify the LLM's causal contribution.
- In 16 paired `(task, seed, policy)` comparisons, 15 action sequences and
  10 reported within-bank risk curves coincide. This supports an action
  influence diagnosis, but equal actions do not imply equal predictions,
  and different within-bank risk curves are not common-target losses.
- Seven of the 16 Full targeted query scores are zero; the reported Pearson
  correlation is `-0.5750`. Removing zero-score observations leaves nine
  positive-score queries with Pearson correlation `-0.5572`. Thus zero-score
  ties alone do not explain the negative association. Realized class risk
  sometimes rises after an observation even under a well-specified model;
  individual increases are not by themselves a model-misspecification test.
- CRK1 has zero initial class risk on its recorded trajectories. Keep this
  registered task in the original denominator; show an explicitly labeled
  informative-risk sensitivity analysis if needed, without replacing the
  frozen metric.

## The proposed 48E+L comparison

`48` is a limit on *unique candidate validation work*, including search and
pruning trials. The No-LLM artifacts use all 48 validation slots but expose
only 15--19 evaluated bank rows per task/seed. The experimental treatment
should therefore be described as `48 baseline evaluations + up to L additional
LLM evaluations`, rather than 48 distinct engine hypotheses.

`quality_first_augmentation.py` constructs the response-free candidate pool:

1. Verify paired development/validation role identities and exact engine
   candidate evidence across both conditions.
2. Preserve the No-LLM evaluated bank verbatim. Require that it used all
   48 baseline evaluations; do not inherit the historical Full's 36-slot
   deterministic path.
3. Check every compiled LLM proposal's parent engine lineage and compiler
   identity; append proposals with a new structural support. Allow at most
   12 compiled proposals per pair in this registered diagnostic. Do not
   claim that compiler acceptance is predictive admission.

The supplied eight pilot pairs contain **17 compiled LLM proposals**;
**14** are new supports relative to the frozen No-LLM evaluated bank or
earlier proposals. One pair has zero compiled LLM proposals. The attached
archive can establish this counterfactual candidate *construction*, not
its independent predictive value or posterior/decision gain.

For a prospective quality-first experiment, the deterministic stage must
execute and log the same 48 validation operations in both arms before LLM
proposals are evaluated. The new L stage needs a separately metered budget,
an explicit failure/abstention policy, and full provenance. Report three
estimands separately:

- **Predictive increment:** independently scored log density and point-loss
  on untouched reporting responses, with baseline and augmented banks
  evaluated on the same rows.
- **Candidate and action increment:** posterior candidate mass under the
  actual production prior, regional predictive difference, and the actions
  each bank nominates under one registered, common decision objective.
- **Realized increment:** external response loss after each action under the
  same task and budget. The archived pilot's within-bank class-risk AULC
  cannot serve as this common external loss.

Compare with an equal-sized non-LLM candidate augmentation in a separate
mechanism control. Record compute and LLM cost. `48+L` is deliberately an
unequal-compute capability ceiling; a favorable result there does not show
matched-resource Full superiority or isolate the LLM's intelligence from
simply adding candidates.

No new formal run is authorized by this audit: a prospective runner must
first implement the second-stage validation budget, preserve the original
No-LLM stage byte-for-byte, fix a common external decision target, and pass
inference and decision-correctness checks. Historical output directories
remain immutable.
