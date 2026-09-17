# Short scientific-system development pilot

Status: registered user-only development pilot, bound to the local ignored
`config/bigmodel.local.json` GLM-5.3 public provider identity. A credential-free
transport preflight returned HTTP 200 from model `glm-5.3` and one protocol-valid
candidate; the key itself is excluded from Git and the registration identity.
The sole configuration is configs/scientific_system_short_pilot.json. A null
public provider identity is permitted only for an unauthorized draft.
There are no placeholder credentials, fabricated results or efficacy claims.

Purpose: bounded end-to-end development screening, not confirmation, superiority
certification or a result-selected rerun of any historical P3M experiment.
One prespecified CCPP seed uses full/no_llm/single_engine exploration and then
same-target EIG/random comparisons within each ablation: six measured policy
runs with four labels each. The original outer sealed partition is unchanged.
The seed is the first historical registered seed, not selected from efficacy.
Exploration, H0 and development evaluation observations have disjoint IDs.

All branches receive two engine jobs, at most 48 unique candidate validations,
one exploration cycle, and the same registered ceilings. In provider-enabled
branches, 16 of those 48 validations are reserved for LLM proposals so that
deterministic initialization cannot silently starve the registered LLM phase;
the no-LLM branch may use all 48 deterministically. Provider-enabled completion
with zero provider attempts is protocol-invalid. Single-engine uses two repeats,
not half the engine work. No knowledge sharing, engine retries, compatibility
filtering, utility fallback or automatic restart is allowed.
Provider transports including retries share the 24-attempt stage cap; no_llm
has a zero transport quota. Numerical ranking must pass at the existing controls
or abort before reveal. Unsupported retained hypotheses reject the entire freeze.

Hard stage caps after the GLM-5.3 latency repair: 30 seconds loading; three
240-second explorations; three
60-second target freezes; six 60-second measured/evaluation policies. The total
isolated-stage ceiling is 1290 seconds (21.5 minutes), not a promised successful
run time. Source checks, evidence publication and process cleanup add overhead.
A timeout is retained as failure, not converted to a passing result. The current
adapter supports only its declared finite, non-evaluating basis library and
Gaussian/NIG refit; this pilot does not establish support for arbitrary symbolic
functions or compositions.

Report every branch, failure, elapsed time and provider attempt. Metrics are
independent opened-development RMSE prefixes and normalized arithmetic mean RMSE,
not historical nAULC or held-out performance. Class targets differ across
exploration ablations and cannot be pooled for class-risk superiority. CCPP-only,
single-seed results cannot justify family-level, statistical or paper superiority.
No automatic efficacy GO threshold is introduced. A protocol-valid negative is
immutable; it cannot be rerun to seek a favorable seed or outcome.

The first user execution at source `542730ffe0e1627f0ad3f8675f99e06f6e98071b`
is immutable protocol-invalid infrastructure evidence. All three exploration
branches completed, but the first measured target freeze rejected retained
degree-four, mixed-degree and `cos(x0)` structures because the discovery grammar
and the downstream closed-basis adapter declared different capabilities. No
measured comparison completed and held-out remained closed. The failed output
must not be resumed or overwritten.

The first GLM-5.3 execution is also immutable protocol-invalid infrastructure
evidence: deterministic exploration finished in milliseconds, but four parallel
provider islands did not finish inside the 120-second stage ceiling. An earlier
response-free provider preflight had already measured a 74-second successful
response after one empty-content attempt, so the ceiling and duplicated network
fan-out were inconsistent before efficacy could be measured. The repaired short
pilot registers one `balanced` island shared by every ablation. Its existing
balanced/Pareto policy jointly audits accuracy, tail error, complexity and
novelty, while reducing provider fan-out to one request batch per enabled branch.
The 240-second ceiling is frozen from transport latency evidence, not efficacy.

That execution also exposed a separate orchestration defect: all 48 candidate
validations were consumed before the LLM phase, so the nominal full branch made
zero provider attempts and duplicated the no-LLM path. The repair reserves a
fixed part of the unchanged total validation budget for the provider phase and
fails closed if an enabled provider is never attempted. This is a stage-budget
contract correction, not a result-driven change to candidates, seeds, metrics,
measurement budgets or acceptance rules.

The source repair does not filter candidates and does not use their validation
performance. It extends the non-evaluating finite basis registry to the
discovery grammar's degree-at-most-four monomials and explicit single-variable
`sin`, `cos` and `tanh` atoms. Fitted discovery amplitudes are still discarded;
all retained structures are refit under the same Gaussian/NIG contract. Unknown
functions, division, composition, degree above four, duplicate/cancelling terms
and malformed tokens still fail closed. A repaired user execution therefore
requires a new clean source freeze and a unique output directory while retaining
the original configuration, data split, seed, budgets, policies and failure.

The runner first reuses the existing production provider file, preserving model,
reasoning, thinking, sampling, timeouts and retries. Invalid saved files fail
closed without environment fallback. Only when the file is absent may the
project-specific environment settings supply the provider. Generic app credentials
are not appropriated. Preflight verifies public provider identity without any
network request; changing endpoint/model/controls requires a new frozen identity.
Only a credential-free public provider-settings hash enters the configuration.
Then authorize the exact configuration, commit it locally, capture clean source,
config, runtime binary and dependency identities, and execute no-data preflight.
No GitHub push, provider request, real data read or experimental run is part of
this registration and freeze preparation.
