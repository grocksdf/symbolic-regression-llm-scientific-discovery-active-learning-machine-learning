# Typed AI-scientist orchestration contract

## Objective

The production LLM is a bounded scientific policy, not another formula source.
It may formulate mechanisms, select registered engine skills, allocate a fixed
job budget, compare engine evidence and issue synthesis instructions.

It may not access acquisition-pool responses, reporting-validation responses,
held-out objects, Bayesian target internals, confirmation outcomes or
experimental authority.

## Production sequence

1. `ProposalRuntime.plan_research` returns one strict JSON `ResearchPlan`.
2. The plan allocates the exact registered engine-job budget across available
   skills.
3. `EngineScheduler.run_allocated` executes those jobs with deterministic
   lineage, timeout, failure and budget accounting.
4. Only expression, development-validation score, complexity, lineage and
   registered engine diagnostics enter `review_engine_evidence`.
5. The typed `ScientistReview` records supported and contradicted mechanisms,
   cross-engine conflicts, synthesis instructions and a stop decision.
6. The existing `ScientificDiscoveryRuntime` receives that immutable context
   while proposing complete executable equations.
7. Independent candidate, source, posterior, utility and transaction gates
   remain authoritative. LLM approval cannot pass a Gate.
8. Research plan, engine evidence and review identities enter the existing
   hash-chained evidence path.

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

This contract is correctness architecture only. It is not efficacy,
superiority or confirmation evidence.
