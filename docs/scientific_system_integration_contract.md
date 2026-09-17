# Scientific system integration contract

## Production path

DiscoveryAgent -> EngineScheduler -> normalize_candidates ->
ScientificDiscoveryRuntime -> HypothesisSpec -> EvidenceRegistry.

All candidate engines share the equation validator and evaluation policy.
Canonical duplicates preserve every engine provenance record. LLM proposals
retain provider attempts, parent hashes and final lineage. System cycle evidence
is appended to the existing verified evidence chain; no second registry exists.

## Evidence roles

Internal validation is used for discovery selection and is not independent
confirmation. System telemetry does not prove efficacy, mechanism recovery or
superiority. System pair validation requires complete full/no_llm/single_engine
coordinates, identical data fingerprints, measurement budget, engine job budget,
candidate evaluation budget and compute ceiling, with failures blocking effect
analysis. Provider-free ablations must record zero calls.

## PCPI boundary — not yet an executable system loop

The current DiscoveryAgent deliberately cannot select/reveal pool labels.
Historical P3 runners remain isolated. Discovery output is a hypothesis proposal,
not automatically a normalized Bayesian prior or a frozen operational class map.
LLM edits after H0 must not replace the scientific target inside a sequential run.

Before an end-to-end PCPI system experiment is authorized, a separate adapter
must freeze a supported prior/likelihood, candidate-to-state representation,
operational estimand, action domain and budget identity from the exploration
phase. Unsupported hypotheses must fail closed, not be silently dropped or
assigned fabricated posterior mass. The actual score-select-reveal-update path
must then pass target and transaction correctness tests.

The first restricted adapter is now implemented in discovery/pcpi_adapter.py.
It supports the existing closed polynomial basis only. The caller must explicitly
authorize discarding fitted discovery coefficients and refitting the support with
the declared Normal--Inverse--Gamma prior and Gaussian observation model. Unique
supports receive a uniform conditional prior; source expressions remain bound in
the frozen identity. Unsupported terms reject the entire freeze. H0 development
responses, action domain and budget-derived class map are separately hash-bound.
Response-selected support is conditional development modelling, not an unconditional
Bayesian discovery claim; independent selection/inference separation is still needed
for that stronger interpretation.

The restricted adapter passes posterior batch/sequential and identity correctness
tests. pcpi/discovery_transaction.py now supplies isolated Gaussian-model
plan/admit/recovery transactions using the existing frozen-class EIG estimator.
It requires a certified ranking before publishing a decision and opening its
matching raw-coordinate indexed response. The registered static PoolOracle is
supported; physical laboratory instruments and arbitrary oracle subclasses are
not. A crash before receipt can re-read the same immutable static label but never
counts that as a second observation. Use a single writer per workspace.
Source, scoring controls, target, candidate matrix and contiguous response prefix
are bound to durable records. Historical workspace identities are rejected.
The budget-loop executor binds a full POOL_CONTRACT, resumes the already-admitted
prefix, removes its selected IDs and publishes a terminal RUN_MANIFEST only after
the frozen measurement budget completes. Completed recovery does not call the
oracle again. Uncertified ranking stops before reveal and has no utility fallback.

This is a separate declared Gaussian likelihood contract, not a substitution
inside historical P3M's semiparametric response model. Historical residual-law
checkpoints cannot be reused. DiscoveryAgent remains exploration-only; the new
transaction layer is not yet connected to a preregistered system experiment runner.
Full system experiment registration, matched random-query comparator, compute
ceilings, fixed seeds/splits and reporting gates remain required. No real
system-loop command is authorized by this integration work. Acquisition-off
must mean matched-budget random queries, never unchanged data or zero queries.

The development coordinator in discovery/system_run.py now freezes proposal
supports and one H0 target once, then runs class EIG and same-budget random
queries through the same durable transaction implementation. Random selection
is query-index seeded and explicitly not rank-certified; its identity cannot
reuse EIG workspaces. Recovery retains its durable selected ID. Both policies
share posterior, class map, candidate pool and numerical ceilings; random need
not expend the EIG ceiling. First failures are retained and block comparison
completion. The analysis entrypoint calls the existing paired-ablation gate
before exposing internal validation rows, without promoting them to efficacy.
The coordinator is now called by the registered system_executor path described
below. Source/config/runtime formal freezing remains required before user execution.

## Matched exploration execution and freeze boundaries

system_ablation.py uses the production agent for full/no_llm/single_engine
exploration. Single-engine repeats equal the total multi-engine job count;
engine retries and shared knowledge are disabled. Candidate-evaluation ceilings,
data fingerprints and seeds are shared. No_llm makes zero provider attempts.
Exploration has measurement_budget=0 honestly; it is not an acquisition-off
comparator. Separate EIG/random transactions retain matched nonzero measurement
budgets. Internal validation remains selection data, never confirmation.

Actual job failures, candidate evaluations, provider attempts and elapsed time
are checked before analysis. In addition, stages run in isolated spawned
processes with a parent wall timer, termination and join on timeout/interruption.
Only the internal polynomial_lasso and mcts engines are supported; external
engine/lab subprocesses are not covered or authorized. ResourceLimits caps wall
time, not RAM or CPU instructions. A shared atomic quota is consumed immediately
before each provider transport, including retries, parallel islands and cycles;
denial blocks completion even if a runtime catches the exception. Candidate
validation retains the production hard evaluation budget.
Interrupted unfinished exploration cannot silently reissue external
calls; completed variants recover from immutable results under the same contract.
The contract stores a public provider-settings hash, never credentials or their
hashes. Completion alone is not efficacy evidence.

system_freeze.py reuses existing clean-Git, binary and dependency checks and binds
the configuration bytes. Dirty source fails before runtime or data access.
system_executor.py and scripts/run_scientific_system.py now compose registered
open-development loading, exploration ablations, all retained proposals, explicit
refit, one frozen H0 target per ablation, and same-budget EIG/random transactions.
Each policy shares its ablation's model, initial posterior, partition, candidate
pool, prior and numerical ceilings. Different exploration ablations may discover
different supports: their class spaces are not claimed comparable, and no pooled
cross-ablation class-risk superiority statistic is calculated. Unsupported
formulas reject the entire protocol; there is no favorable compatibility filter.
The registration declares seeds, counts, prior, likelihood, numerical schedule,
public provider identity and stage budgets. Discovery environment overrides are
forbidden. Outputs must live outside the frozen source root. Clean Git/runtime
identities are checked before data and between stages. A terminal failure blocks
automatic resumption; completed measured policies recover without new labels.
Experiment evidence is appended to the existing verified hypothesis registry,
not a second registry. Public credential-free identities survive key rotation.

data/system_protocol.py reads Gas CSVs for 2011--2014 only, at explicit filenames
with mandatory official hashes; no sealed year file is located, hashed or opened.
For CCPP, the historical outer SHA256 partition at SPLIT_SEED is preserved. Only
registered open-row cells are decoded from the source XML stream; excluded
responses are never converted, retained, summarized or exposed. Official source
integrity requires reading container bytes; this is not held-out evaluation.
Counts and within-open-role order are target-blind and preregistered. Exploration,
inference H0 and independent development evaluation use disjoint observation IDs.
The inference stage never receives exploration validation as its metric data.

After durable measurements, reporting reconstructs each prefix and records RMSE
and normalized arithmetic mean RMSE on the independent opened development
evaluation role. This is explicitly not historical nAULC or sealed confirmation.
No favorable seed/dataset filtering or automatic superiority assessment occurs.
Both Gas outcomes retain one family identity. Registration/freeze preflight opens
no datasets; execution additionally requires an authorized config and user-only
entrypoint. The correctness gate validates wiring, boundaries and hard limits;
it cannot substitute for a clean-source freeze or genuine efficacy evidence.

## Historical evidence (unchanged)

All P3M/DCCA outputs remain immutable negative development evidence. No gate
correction can certify efficacy or prove counterfactual choices using unobserved
responses. The interrupted monotone DCCA experiment was withdrawn, not registered
as a successful new algorithm.
