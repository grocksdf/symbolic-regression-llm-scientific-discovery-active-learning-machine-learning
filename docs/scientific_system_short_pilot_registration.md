# Short scientific-system development pilot

Status: registered user-only development pilot, bound to the existing production
config/bigmodel_glm_5_2.json public provider identity. No transport request has
been made during registration; key presence is not proof of authentication.
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

All branches receive two engine jobs, 48 unique candidate validations, one
exploration cycle, and the same registered ceilings. Single-engine uses two
repeats, not half the engine work. No knowledge sharing, engine retries,
compatibility filtering, utility fallback or automatic restart is allowed.
Provider transports including retries share the 24-attempt stage cap; no_llm
has a zero transport quota. Numerical ranking must pass at the existing controls
or abort before reveal. Unsupported retained hypotheses reject the entire freeze.

Hard stage caps: 30 seconds loading; three 120-second explorations; three
60-second target freezes; six 60-second measured/evaluation policies. The total
isolated-stage ceiling is 930 seconds (15.5 minutes), not a promised successful
run time. Source checks, evidence publication and process cleanup add overhead.
A timeout is retained as failure, not converted to a passing result. The current
adapter supports only its declared closed polynomial library and Gaussian/NIG
refit; this pilot does not establish support for general symbolic functions.

Report every branch, failure, elapsed time and provider attempt. Metrics are
independent opened-development RMSE prefixes and normalized arithmetic mean RMSE,
not historical nAULC or held-out performance. Class targets differ across
exploration ablations and cannot be pooled for class-risk superiority. CCPP-only,
single-seed results cannot justify family-level, statistical or paper superiority.
No automatic efficacy GO threshold is introduced. A protocol-valid negative is
immutable; it cannot be rerun to seek a favorable seed or outcome.

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
