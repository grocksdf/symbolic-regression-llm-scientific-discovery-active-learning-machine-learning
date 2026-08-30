# PCPI P3I.3 supervised real-development execution protocol

## Status and scope

P3I.3 freezes one user-executed, held-out-closed real-development comparison.
It follows the immutable negative P3H.9 result and the response-free P3I.1 and
P3I.2 repairs. It is failure-informed development evidence, not independent
confirmation. Codex is not authorized to execute it.

The P3I.3 scientific and computational configuration is byte-for-byte equal to
the P3I.2 candidate after removing only `schema`, `stage` and
`operational_execution_authorized`. P3I.3 changes those three execution identity
fields only. It does not change a dataset, seed, budget, likelihood-power family,
class threshold, numerical schedule, safe-set rule, utility, assessment boundary
or tie break after observing results.

## Frozen C1--C3 contract

All four policies share the registered datasets, eight seeds, 16 base-warmup
and 16 residual-training observations, 32 acquisition responses, 128-candidate
pool, 256 validation observations, eta-one reporting posterior and the same
initial-frozen decision-regret class partition. The partition uses posterior
mean profiles and one common H0 mixture scale at the unchanged budget-derived
threshold.

PCPI alone receives the complete likelihood-power family
`(0.125, 0.25, 0.5, 1.0)` and each model's matching prequential residual state.
It ranks the covariate-only representative-safe set by

`min_eta I_eta(C; Y_a)`.

The residual marginal is represented by the registered common invertible PIT
transport, which preserves class mutual information. Conditional predictive
EPIG is exactly zero. No Student-t joint utility, posterior variance, QBC,
minimum-MMD fallback or response-fitted switch is admissible.

The primary assessment is paired frozen-class entropy-gain improvement versus
random in both registered dataset families, together with controlled class-gain
negative transfer, nondegenerate class aggregation and a valid decision rule.
Validation RMSE and NLL are secondary reported limitations and cannot change the
primary class-risk decision.

## Supervised execution and failure visibility

The only admissible launch route is `scripts/invoke_pcpi_p3i3_supervised.ps1`
with the final frozen branch, commit, tree and config SHA-256 supplied explicitly.
Before starting Python it verifies all four identities, requires a clean Git
source, permits only the user's historical untracked `evidence/`, moves that
directory to an explicit checked sibling path, and rejects any pre-existing
output, console log or stash path.

The same launcher has a `-PreflightOnly` mode. It exercises the final identity,
clean-tree and evidence-isolation/restoration path, then returns before output
creation or child-process launch. Codex may execute only this no-data mode; the
real mode remains user-only.

The supervisor starts the frozen runner in a hidden child process and displays
the latest fsync-backed progress event every time it changes. Stdout and stderr
are preserved beside the unique output. The first runner failure creates
`TERMINAL_FAILURE.json` and aborts immediately. A nonzero child exit prints the
last stderr records. User interruption stops the child before the supervisor
restores historical evidence.

Success requires all 96 policy runs, zero failures, a passed protocol Gate,
closed held-out, no held-out selection, a valid evidence registry, and exact
manifest matches to the supplied source commit, tree and config hash. An
efficacy-negative but protocol-valid terminal result is preserved as negative
evidence and must not be rerun.

## Correctness Gate and claim boundary

The no-data P3I.3 Gate passes all eleven registered decisions. The focused
P3I/P3H/integrity regression passes all 55 tests, and the PowerShell supervisor
passes parser validation. The complete repository regression passes `764/764`
with final exit code zero, and `git diff --check` passes. No historical test,
failure or assertion is masked or relaxed. These checks access no simulated
experiment, real data, validation response, candidate response or held-out
value.

P3I.3 may establish only the preregistered real-development class-risk result.
It cannot establish universal superiority, independent confirmation, physical
intervention, a scientific law, open-grammar discovery, held-out performance or
paper acceptance.
