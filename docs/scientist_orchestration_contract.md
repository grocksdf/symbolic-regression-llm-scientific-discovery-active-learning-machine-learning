# Typed AI-scientist orchestration contract

## Objective

The production LLM is a bounded scientific policy, not another formula source.
It may formulate mechanisms, select registered engine skills, allocate a fixed
job budget, compare engine evidence and issue synthesis instructions.

Under the typed evidence-synthesis profile, those synthesis instructions are
not executable equation text.  They are lineage-bound directives over
registered operations (`UNION_SUPPORTS`, `INTERSECTION_SUPPORTS`, or
`AUGMENT_BASE`).  The deterministic compiler resolves the referenced engine
expressions, constructs one closed-basis support, and leaves all coefficients
to the existing global refit.  Unknown lineage, duplicate parents, unsupported
operations, unchanged support and adapter-incompatible output fail closed.

It may not access acquisition-pool responses, reporting-validation responses,
held-out objects, Bayesian target internals, confirmation outcomes or
experimental authority.

## Production sequence

1. `ProposalRuntime.plan_research` returns one strict JSON `ResearchPlan`.
2. The plan allocates the exact registered engine-job budget across available
   skills.
3. `EngineScheduler.run_allocated` executes those jobs with deterministic
   lineage, timeout, failure and budget accounting.  A call that requests
   several registered operations is compiled into a job-level evidence
   matrix: singleton operation jobs are executed first, followed by the
   complete requested operation set when budget remains.  Jobs may not all
   receive the same union of operations, because that would erase the
   Scientist's requested comparison.
4. Only expression, development-validation score, complexity, lineage and
   registered engine diagnostics enter `review_engine_evidence`.
5. The typed `ScientistReview` records supported and contradicted mechanisms,
   cross-engine conflicts, synthesis instructions and a stop decision.
6. A bounded `ScientistState` records engine summaries, surviving hypotheses,
   plan/review identities and cumulative job use. The next round must receive
   this state before issuing a new plan.
7. The existing `ScientificDiscoveryRuntime` receives that immutable context
   while proposing complete executable equations.
8. Independent candidate, source, posterior, utility and transaction gates
   remain authoritative. LLM approval cannot pass a Gate.
9. Research plan, engine evidence, state transition and review identities enter the existing
   hash-chained evidence path.

Registered cycle count remains an exact matched-compute contract. A Scientist
may request stopping, but formal ablations do not silently save compute for one
variant; the request is recorded and the remaining registered rounds continue.

## Task-local structural memory

The production closed loop may stage a validated LLM lineage for later rounds
of the same frozen task namespace.  This memory is development-only: it is
validated before retrieval, records its stage identity and failure signature,
and is exposed to later proposal prompts as
`task-local-development-staged`.  It cannot enter the reusable cross-task
structure library, cannot change the registered compute budget, and never
contains acquisition-pool, reporting-validation or held-out responses.

Reusable cross-task knowledge remains a separate capability.  Promotion still
requires a verified passing independent untouched confirmation.  Enabling
task-local memory therefore does not enable `use_knowledge`, weaken promotion,
or create a confirmation claim.

The paper-faithful production profile uses multiple registered outer cycles,
multiple structural objective islands and multiple bounded inner refinement
rounds.  Every value is frozen before data access.  A no-LLM ablation receives
the same engine and candidate-evaluation budget but cannot read or write LLM
lineage memory because it has no provider proposal phase.

Evidence-conditioned synthesis is a non-destructive portfolio family.  The
fixed seed compiler preserves one frontier member from every registered engine,
the previous survivor and the deterministic anchors before admitting synthesized
lineages.  Synthesis may add candidates within the remaining budget but cannot
evict those coverage roles.  This is an algorithmic resource invariant, not a
dataset-specific repair.

## Inference routing

Finite frozen hypothesis banks use the exact conjugate posterior. An open or
transdimensional bank may request certified SMC only through the inference
router and only after a separate SMC integration authorization. The current
scientific-system runner accepts `auto` or `exact_finite`; unauthorized open
SMC fails before response access.

## Skills

The initial skill registry contains:

- `polynomial_lasso`: fast sparse polynomial baseline and global interactions;
- `mcts`: typed nonlinear structural search and diversity frontier.

Skills declare capabilities and inductive bias. They do not expose arbitrary
Python execution or direct data handles to the LLM. Future PySR, literature,
simulation and laboratory tools must implement the same capability, cost,
permission and evidence contract before registration.

## Ablations

- `full`: LLM research plan, multi-engine dispatch, evidence review and
  hypothesis synthesis.
- `no_llm`: identical engine, candidate, compute and measurement budgets with
  deterministic engine allocation and deterministic evidence review.
- `single_engine`: the same LLM scientist policy and total engine-job budget,
  but only one engine skill is available.

These are policy-level system ablations. They must not be replaced by forcing
all engines to receive posterior mass on every task.

## Future runtime boundary

The typed Python controller is the scientific source of truth. A later
LangGraph layer may host the cognitive plan/review/reflection graph. Temporal
may provide durable formal execution, and MCP may expose external skills.
None of those frameworks may own the posterior, utility, held-out capability
or evidence verdict.

## Cross-task policy learning

Historical response-free candidate/source certificates may train a Bayesian
skill-reliability prior only through leave-one-task-out replay. Each task
contributes at most one Bernoulli outcome per skill; arbitration folds certify
that outcome but are not counted as independent replicates. A Jeffreys
Beta(1/2, 1/2) prior supplies the reliability posterior.

Production use requires at least three training task identities from at least
two registered dataset families for every held-out replay task. Until that
coverage Gate passes, Scientist planning receives no learned reliability prior.
Acquisition responses, reporting-validation responses and held-out outcomes are
never inputs to policy replay.

The September 20, 2026 artifact replay found seven task coordinates but only
one `gas_turbine` family. It is retained as
`insufficient-cross-family-coverage`, not as skill-performance evidence.

This contract is correctness architecture only. It is not efficacy,
superiority or confirmation evidence.
