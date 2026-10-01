# Posterior-gap-directed Full: exploration implementation and boundaries

Status: **new development exploration path, no measured-pool authorization**.
The historical Full/No-LLM failure and all old protocol identities stay
unchanged. `execute_registered_system` still refuses typed augmentation.

The optional `DiscoveryAgentConfig.posterior_gap_directed=True` works through
the existing `run_exploration_ablations` Full / No-LLM / Single flow. Its
`discovery_budget` is now the protected engine-only reference budget (for
example 48 evaluation slots; the number of distinct engine hypotheses is
reported separately and need not equal 48). Full receives that budget plus its registered typed and inner
LLM reserves; No-LLM receives the entire engine-only budget. The engine
schedule is frozen before LLM review, and identical raw engine frontiers are
checked and restored into both candidate lists. The single-engine control
gets the same total evaluation ceiling as Full. Additional provider time and
candidate admission work must be recorded separately; this is a
**quality-first, unequal-compute** comparison, not matched-compute efficacy.

Inputs in addition to the existing exploration arguments:

* `gap_audit`: independent validation responses for diagnosing inadequate
  frozen engine predictive intervals;
* `gap_admission`: a second, disjoint validation role for optional-candidate
  predictive evidence;
* `gap_prior`: the predeclared `NormalInverseGammaPrior`;
* `gap_measurement_budget`: registered budget used to freeze the provisional
  engine-bank operational classes.

Both response roles must be row-disjoint from development and from discovery
validation, and from each other. The existing benchmark discovery validation
rows are already consumed; they cannot be repurposed as either role. The
acquisition pool contributes **covariates only** for the target domain and a
covariate-frozen median split on axis zero. No pool/held-out response is read.

Flow per cycle:

1. Run the same engine allocation in Full and No-LLM. Freeze an *interim*
   engine bank using the registered closed basis and development H0.
   Disagreement is a descriptive diagnostic. The posterior is conditional on
   the development-time engine search, not a posterior over all possible laws.
2. Before any gap-directed provider review, test the existing bank on
   `gap_audit`. For each covariate-only region, count outcomes outside the
   frozen 90% Student-t mixture predictive interval. A simultaneous
   Bonferroni one-sided binomial lower bound must exceed the nominal 10%
   miss rate. If none passes, skip both Scientist review and inner LLM
   proposals for that cycle and retain the engine bank. This detects one
   concrete form of **independent predictive inadequacy**; it is not a
   complete test for missing symbolic structures and depends on the declared
   predictive calibration null.
3. Supply the posterior class contrast, bounded region, independent miss
   counts, and top existing structures to the existing proposal payload. A
   typed candidate may include `correction: g(y_hat,x)` instead of `equation`.
   `ProposalRuntime` substitutes its exact current incumbent, expands under
   finite AST/term budgets, checks the closed basis and compares the
   *materialized* support against the frozen engine bank. No unsupported
   compound function or raw `y_hat` enters `EvaluationRuntime`.
4. After discovery, screen every LLM support on `gap_admission` as a fixed
   conditional predictive expert versus the frozen engine mixture. The
   region selector has equal declared core/optional prior probabilities;
   per-region log predictive Bayes factors and modular selector masses are
   recorded. Its threshold `log(attempted_candidates * eligible_regions /
   .05)` is a simultaneous fixed-law evidence-ratio screen. This is **not**
   the original PCPI symbolic posterior. If the core law is misspecified, the
   threshold has no frequentist type-I interpretation. No candidate is
   admitted on disagreement or the gap screen alone. `D_admit` is consumed
   only after all proposals and engine outputs have been frozen.
5. Existing `KnowledgeRuntime` may stage an accepted correction during
   discovery. The subsequent independent admission outcome is attached to
   that same stage. An admitted stage gets status
   `prediction_admitted_pending_decision`, remains outside task-local and
   cross-task retrieval, and cannot pass existing promotion. Its record
   retains target/admission identities, region selector evidence, and the
   materialized expression, attempted count, region identity, rejection
   reason, and unresolved action-effect flag. No positive *decision* lesson is stored
   until an independent common-target decision effect is verified.

Important limitations:

* The Full candidate list passed to the analysis includes all intact engine
  supports plus independently admitted LLM supports. Downstream
  `select_operational_capacity_bank` can still prune supports under an old
  final-bank capacity. The measured executor explicitly blocks this mode;
  a new source-projection, frozen common-target loss, independent reporting
  split, candidate budget and transaction-replay gate are required first.
* The region test is deliberately narrow: well-calibrated average intervals
  can hide systematic mean bias, and low-count regions usually abstain.
  There is no implied universal Full > No-LLM guarantee.
* The action contribution under plausible real laws is **not** inferred
  from candidate selector mass. Until a same-target law family has been
  independently calibrated and the action-level gate passed, the code logs
  `decision_contribution_assessed=False` and permits no measured policy
  change.

Correctness check on the user's Windows source checkout (no real data or
provider call):

```powershell
& 'D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe' -m pytest -q 'D:\01\666\hypothesis_mvp\tests\test_posterior_gap_full_pipeline.py'
```

This path is intentionally opt-in and requires predeclared disjoint
`gap_audit` and `gap_admission` role datasets. No real-data execution command
is provided before a registered independent split and common-target Gate.
