# Posterior-gap-directed Full: exploration implementation and boundaries

Status: **new development exploration path, no measured-pool authorization**.
The historical Full/No-LLM failure and all old protocol identities stay
unchanged. `execute_registered_system` still refuses typed augmentation.

The optional `DiscoveryAgentConfig.posterior_gap_directed=True` works through
the existing `run_exploration_ablations` Full / No-LLM / Single flow. Its
`discovery_budget` is now the protected engine-only reference budget (for
example 48 evaluation slots; the number of distinct engine hypotheses is
reported separately and need not equal 48). Full receives that budget plus
its registered typed and inner LLM reserves; No-LLM receives the entire
engine-only budget. The engine
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
* Optional `decision_reference`: a `FrozenFiniteActionReference` with the
  same covariate target and finite response grid for every action, fixed
  target weights, and explicitly supplied plausible finite laws;
* Optional `decision_calibration`: its disjoint validation role, bound to the
  reference by fingerprint. These two inputs must appear together.

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
   `gap_audit`. Each covariate-only region has four prespecified checks:
   predictive interval undercoverage; excess observations above the frozen
   predictive median; excess observations below it; and a positive regional
   log-score likelihood ratio for a frozen development-only generic anchor
   reference against the engine mixture. Divide `.05` across four checks and
   every frozen region. The first three use exact one-sided binomial bounds;
   the fourth uses a likelihood-ratio threshold under a declared product-core
   null. If no check passes, skip both Scientist review and inner LLM
   proposals for that cycle. Directional PIT imbalance can flag a persistent
   location shift within wide intervals, but it is not a complete test for
   mean bias; any of these model-conditional cues may also be wrong under
   dependent observations, covariate shift, or a misspecified reference law.
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
5. Project the Full exploratory bank to the **exact No-LLM support baseline**
   plus independently admitted LLM supports. Freeze both complete banks on
   the same development responses and covariate domain, without dispatching
   the old `maximum_candidates` selector. Record identities and support
   counts in `QUALITY_FIRST_EXPANDED_BANKS.json`. This resolves final-bank
   crowd-out within this exploratory comparison. The resulting class
   partitions are still bank-dependent; a shared domain is not a common
   class-loss or measured-policy certificate. The frozen diagnostic models
   use their declared support-count source prior, not independently
   arbitrated source weights for a new measured protocol.
6. If an independent finite action reference and its calibration provenance
   are supplied, reconstruct both fixed experts, update the regional selector
   on `gap_admission`, and use the existing
   `candidate_action_contribution` calculation to compare actions with and
   without each admitted candidate under **the same** finite laws and squared
   target loss. Retain signed per-law effects even when negative. This is a
   conditional diagnostic, not a numerically certified continuous integral
   or an audit of PCPI's registered operational-class decision. Its
   `pcpi_operational_class_decision_assessed` and
   `decision_contribution_assessed` flags remain false, and
   `finite_law_action_contribution_assessed` reports the narrower calculation.
7. Existing `KnowledgeRuntime` may stage an accepted correction during
   discovery. The subsequent independent admission outcome is attached to
   that same stage. An admitted stage gets status
   `prediction_admitted_pending_decision`, remains outside task-local and
   cross-task retrieval, and cannot pass existing promotion. Its record
   retains target/admission identities, region selector evidence, and the
   materialized expression, attempted count, region identity, rejection
   reason, and unresolved action-effect flag. No positive *decision* lesson is
   stored until an independent common-target decision effect is verified.

Important limitations:

* `select_operational_capacity_bank` and the measured executor retain their
  old fixed capacity behavior; neither consumes this exploratory expanded
  bank. The measured executor explicitly blocks this mode. Opening it needs
  new source projection, a frozen common-target loss, independent reporting
  split, candidate budget, numerical error budget and transaction-replay gate.
* The multi-signal region test remains a model-conditional search permission.
  Broad calibrated intervals can still conceal some mean or structural
  errors; small regions normally abstain. This is not a universal Full >
  No-LLM guarantee.
* The action contribution under plausible real laws is **not** inferred
  from candidate selector mass. Caller-supplied finite laws must be
  calibrated independently. Even with the optional finite reference audit,
  law coverage and continuous numerical ranking are unproved. The code keeps
  `decision_contribution_assessed=False` and permits no measured policy
  change. Reporting outcomes must remain a separate, untouched role.

Correctness check on the user's Windows source checkout (no real data or
provider call):

```powershell
& 'D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe' -m pytest -q 'D:\01\666\hypothesis_mvp\tests\test_posterior_gap_full_pipeline.py'
```

This path is intentionally opt-in and requires predeclared disjoint
`gap_audit` and `gap_admission` role datasets. No real-data execution command
is provided before a registered independent split and common-target Gate.
