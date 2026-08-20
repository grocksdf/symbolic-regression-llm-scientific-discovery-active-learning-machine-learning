# P3F.4-CERT.7 resident local/RJ source-composition contract

Status: **ACTUAL REJUVENATION SOURCE COMPOSITION PASSES; SMC EXECUTION BLOCKED**

GitHub baseline: `09b383061178b98cd66f9e2aaf1ea28d323c0e50`

CERT.7 imports the CERT.5 involutive proposal and CERT.6 common-target adapter
into an actual resident `_rejuvenate` branch. It performs only response-free
source, exact endpoint, and finite-transition checks. It does not call
`ScalableOpenTargetSMC.run`, materialize responses, or execute simulated,
formal, real, validation, acquisition, confirmatory, or held-out experiments.

## 1. Registered resident source path

The only new proposal identity is:

```text
proposal_kind = raw-state-local-rj
maximum_nodes = None
rejuvenation_population_mode = terminal-only
```

A finite node cutoff is rejected because it would change the proposal and
target support. Waste-free, Rao--Blackwellized, or other population reuse
modes are rejected in this Gate because their composition with the new chain
has not been separately proved. Multiple ordinary terminal-only steps remain
a composition of invariant kernels.

`ScalableOpenTargetSMC.__init__` binds one target contract to:

- the CERT.5 exact local/RJ plan;
- the CERT.6 resident common-target plan; and
- the CERT.7 source-composition identity.

Cross-contract or cross-grammar plan binding fails closed.

## 2. Actual `_rejuvenate` delegation

For a current resident particle representing `z=(T,d)`, the new branch:

1. maps its exact expression/component fields to `RawStateLocalRJState`;
2. samples a CERT.5 address, complete open subtree, and component using the
   arbitrary-precision byte source;
3. constructs the proposed resident particle from the proposed raw endpoint
   through the CERT.6 canonical semantic design;
4. reconstructs the same collapsed marginal state at the current bridge; and
5. delegates target mass, reverse auxiliary mass, proposal ratio and MH
   acceptance to `evaluate_resident_particle_local_rj_transition`.

The branch does not reuse the old independence or uniform proposal-ratio
formula. Its diagnostic record carries the exact site index/path, local/RJ
move type, proposal log ratio and unit log-Jacobian.

## 3. Endpoint target identity

For each endpoint, the resident particle supplies only its semantic identity
and collapsed log marginal to the adapter. The helper checks

\[
\log p_{\rm resident}(T,d)+\log m_{\rm resident}(\kappa(T),d)
=
\log p_G(T)+\log p_D(d)+\log m_{\rm adapter}(\kappa(T),d)
\]

within the pre-registered CERT.4/CERT.5 `2e-12` floating identity tolerance.
The raw/component endpoints, plan hashes, proposal support and prior masses
remain exact discrete identities.

If current and proposed raw ASTs have the same `(polynomial_key, component)`,
their collapsed log marginals must be bit-identical because CERT.6 gives them
the same design and sufficient-statistic operations. Any alias disagreement,
component-field mismatch, endpoint mismatch, non-finite mass, altered prior,
or cross-target plan fails closed before acceptance is evaluated.

## 4. Proposal ratio and invariant flow

For the realized auxiliary path,

\[
q_f=\frac1{|T|}p_G(S)p_D(d'),\qquad
q_r=\frac1{|T'|}p_G(T_a)p_D(d),
\]

and the resident branch uses the adapter result

\[
\log\alpha=
\min\{0,\log\gamma(z')-\log\gamma(z)+\log q_r-\log q_f\}.
\]

The implementation records `log_abs_jacobian=0` and never recomputes or
overrides this value in `_rejuvenate`.

The proof construction is the same discrete involutive RJ identity established
from Green (1995), iMCMC, and the task-independent subtree-replace pattern:

- https://people.maths.bris.ac.uk/~mapjg/papers/RJMCMCBka.pdf
- https://proceedings.mlr.press/v119/neklyudov20a.html
- https://proceedings.mlr.press/v202/saad23a.html
- https://github.com/probsys/AutoGP.jl

No grammar, likelihood, empirical setting, truncation, regularizer, threshold,
seed, response, or result direction is transferred from these sources.

## 5. Finite composed transition proof

The CERT.7 matrix audit enumerates a finite censoring of the same resident
endpoint helper. The finite auxiliary shell contains every subtree that can be
discarded from the audit state shell. Proposals leaving the audit state shell
are censored to self; the production proposal remains complete and untruncated.

Every interior event is evaluated through the exact function called by the
resident source branch. The resulting matrix passes:

- row stochasticity;
- bidirectional interior support;
- detailed balance;
- target stationarity; and
- the common resident/adapter endpoint-mass identity.

All event probabilities and support identities use exact `Fraction`; only
log-flow and matrix residuals use the already registered `2e-12` tolerance.

## 6. Response-free Gate

The CERT.7 runner retains all 29 CERT.3--CERT.6 checks and adds seven checks:

| Obligation | Check |
|---|---|
| Configuration | only full-open terminal-only source composition is accepted |
| Endpoint balance | actual resident endpoints preserve exact ratio and augmented flow |
| Failure closure | semantic, component, prior and adapter mass mismatches are rejected |
| Source delegation | `_rejuvenate` calls the exact proposal, endpoint constructor and adapter helper |
| Execution guard | `run()` blocks before data validation and particle sampling |
| Finite composition | actual endpoint-helper matrix is stochastic, reversible and invariant |
| Identity binding | cross-target plan composition is rejected |

The registered identity is `36/36`; every repository Python file outside
`.git`, `.venv`, and `evidence` is syntax checked.

## 7. Authorization boundary

CERT.7 establishes:

```text
resident_rejuvenation_import_verified = true
actual_rejuvenate_delegation_verified = true
resident_endpoint_mass_identity_verified = true
finite_composed_transition_invariance_verified = true

resident_smc_integration_authorized = false
resident_smc_invoked = false
```

`P3F4_RESIDENT_LOCAL_RJ_RUN_AUTHORIZED` remains hard-coded `False`; the guard
is evaluated before data validation, target access, cache mutation, initial
particle sampling, bridging, resampling, or rejuvenation. CERT.7 therefore
authorizes the source-composed resident kernel but not resident SMC execution.

The next admissible phase is a response-free proof of the full-open
Feynman--Kac bridge/path and mixing contract, including fail-closed population
relative-ESS/path certification. Predictive calibration, real data,
acquisition, held-out access, confirmation, efficacy, discovery and paper
superiority claims remain blocked.
