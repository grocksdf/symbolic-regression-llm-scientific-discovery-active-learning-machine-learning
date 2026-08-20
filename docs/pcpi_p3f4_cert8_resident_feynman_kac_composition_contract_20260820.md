# P3F.4-CERT.8 resident Feynman--Kac common-target composition contract

Status: **RESPONSE-FREE SOURCE COMPOSITION PASSES; RESIDENT SMC EXECUTION BLOCKED**

GitHub baseline: `58d2581b853605603056629b90ef084387a12c13`

CERT.8 proves that the actual resident source path uses one immutable target
identity for analytic bridge selection, incremental weighting, unbiased
resampling and the CERT.7 local/RJ rejuvenation kernel. It performs only
source, algebraic and finite-state combinatorial checks. It does not call
`ScalableOpenTargetSMC.run`, materialize responses, or execute simulated,
formal, real, validation, acquisition, confirmatory or held-out experiments.

## 1. Blocker removed

The historical resident branch chose bridge powers from empirical particle
CESS. It also contained a terminal branch that could force beta to one when
the remaining interval was below a numerical tolerance even if CESS was below
the registered floor. That path was acceptable only as a diagnostic SMC
implementation; it was not the P3F.4 population-relative-ESS certificate.

CERT.8 adds a distinct resident mode:

```text
proposal_kind = raw-state-local-rj
maximum_nodes = None
tempering_mode = certified-population-relative-ess
certification_maximum_nodes >= reference_slice_maximum_nodes
certified_beta_grid_denominator = 32
cess_target_fraction = 0.8
maximum_bridge_steps = 64
resampling_kind = systematic
resampling_schedule = post-bridge
rejuvenation_population_mode = terminal-only
```

Any different runtime configuration fails before data validation. The run
authorization constants remain false, so even this registered configuration
cannot execute resident SMC in CERT.8.

## 2. Common full-open bridge target

At observation `t`, with the first `t-1` observations at power one and the
current observation at beta, define

\[
\gamma_{t,\beta}(z)=p_0(z)m_{t,\beta}(z),\qquad
\pi_{t,\beta}(z)=\gamma_{t,\beta}(z)/Z_{t,\beta}.
\]

Every beta target has the same complete raw `(T,d)` support. Beta changes only
the numerical path; beta one is the ordinary Bayesian update. No cutoff,
generalized likelihood power or empirical particle statistic changes the
terminal target.

`ResidentFeynmanKacPlan` binds the target-contract hash, CERT.7 source-
composition hash, semantic certification cutoff, exact rational beta grid,
population floor, bridge budget, resampler and rejuvenation population mode.
Cross-contract binding fails closed.

For every accepted bridge, `ResidentFeynmanKacBridgeTarget` stores exact beta
numerators and hashes of the current, proposed and second-moment normalizer
certificates. Its stable hash is the sole bridge identity consumed downstream.

## 3. Analytic path selection

For beta less than beta-prime, the incremental potential is

\[
G(z)=\frac{m_{t,\beta'}(z)}{m_{t,\beta}(z)}.
\]

Population relative ESS is

\[
r_{\mathrm{ESS}}
=\frac{\pi_{t,\beta}(G)^2}{\pi_{t,\beta}(G^2)}
=\frac{Z_{t,\beta'}^2}
       {Z_{t,\beta}Z_{t,2\beta'-\beta}}.
\]

The response-energy semantic core and analytic tail give

\[
\underline r(\beta,\beta')=
\frac{Z_{J,t,\beta'}^2}
{\overline Z_{J,t,\beta}\,
 \overline Z_{J,t,2\beta'-\beta}}
\le r_{\mathrm{ESS}}.
\]

The selector evaluates the fixed grid `{0,1/32,...,1}` and chooses the largest
next beta with `underline r >= 0.8`. If no positive step passes, or 64 steps
are exhausted before one, it returns NO-GO. There is no tolerance-sized forced
terminal step.

Empirical CESS remains recorded as a descriptive population diagnostic. It is
computed only after analytic selection and cannot influence the bridge.

The selector accepts exactly the observed response prefix through the current
row. Supplying later response coordinates is rejected. The frozen action grid
may remain available for the response-free structure-wise projection, while
the absent future targets are algebraically padded only at zero likelihood
power.

## 4. Incremental weights and evidence telescope

For a resident particle at state `z_i`, the update is exactly

\[
\ell_i=\log m_{t,\beta'}(z_i)-\log m_{t,\beta}(z_i),
\qquad
W_i'\propto W_i\exp(\ell_i).
\]

`ResidentFeynmanKacWeightUpdate` carries the same bridge hash, current-target
hash and next-target hash. The returned log normalizer increment is

\[
\log\sum_i W_i\exp(\ell_i),
\]

and consecutive increments telescope to the direct endpoint importance
ratio. There is no independent weight formula for the resident branch.

## 5. Resampling and rejuvenation composition

Systematic resampling receives only the normalized weights owned by the
bridge update. Before sampling ancestors, the source revalidates the plan,
bridge, source target, next target and beta identities.

For exact finite weights, randomized systematic resampling is enumerated over
all intervals of its single uniform offset. The combinatorial proof establishes

\[
E[N_j\mid W]=N W_j.
\]

Thus resampling is target-unbiased in empirical-measure expectation. It is not
misrepresented as an exact posterior draw or as a finite-particle accuracy
certificate.

The same bridge and weight-update objects are passed into `_rejuvenate`.
Before any local/RJ proposal is sampled, `_rejuvenate` verifies that beta and
the next-target hash match the weight/resampling target. CERT.7 then owns the
endpoint target masses, forward/reverse proposal ratio and acceptance value.

For a finite target `pi`, exact systematic resampling followed by an invariant
kernel `K` satisfies

\[
E[\widehat\pi_{\mathrm{after}}]
=\pi K=\pi.
\]

The CERT.8 finite proof checks this identity with exact `Fraction` arithmetic.
The actual censored resident local/RJ matrix is separately reconstructed
through the CERT.7 endpoint helper and passes row normalization, stationarity,
irreducibility, positive self-transition probability and a positive finite
spectral gap.

## 6. Source and theory review

The implementation follows the common-space Feynman--Kac separation of target,
potential, resampling and invariant move kernels in:

- Del Moral, Doucet and Jasra, *Sequential Monte Carlo Samplers*:
  https://academic.oup.com/jrsssb/article-abstract/68/3/411/7110641
- Naesseth, Lindsten and Schön, *Elements of Sequential Monte Carlo*:
  https://arxiv.org/abs/1903.04797

The analytic adjacent-distribution condition and its dependence on Markov-
kernel mixing use the 2025 revision of Marion, Mathews and Schmidler:

- https://arxiv.org/abs/1807.01346

Direct divergence control and the distinction between ESS diagnostics and
sampling guarantees follow Huggins and Roy:

- https://arxiv.org/abs/1503.00966

For source-level comparison, current public SMC implementations such as
TensorFlow Probability and `particles` use empirical ESS to adapt tempering:

- https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/mcmc/sample_sequential_monte_carlo.py
- https://github.com/nchopin/particles

Those implementations are useful engineering references, but their empirical
ESS rule is not substituted for PCPI's registered analytic lower certificate.
Recent optimized or persistent SMC population-reuse methods are likewise not
imported because CERT.8 proves only ordinary terminal-only resample-move
composition.

## 7. Response-free Gate

CERT.8 retains all 36 CERT.3--CERT.7 checks and adds ten checks covering:

1. the only registered full-open resident controls;
2. largest certified rational-grid bridge selection;
3. fail-closed behavior with no forced terminal step;
4. incremental-potential and normalizer telescoping;
5. exact systematic-resampling unbiasedness;
6. finite resample-move invariance;
7. actual resident local/RJ irreducibility, aperiodicity and spectral gap;
8. actual source threading of one bridge identity through every operation;
9. pre-sampling and pre-data execution guards; and
10. rejection of cross-target or incomplete plans.

The registered identity is `46/46`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

## 8. Authorization boundary

CERT.8 establishes:

```text
analytic_population_relative_ess_path_verified = true
current_response_prefix_only_verified = true
bridge_target_identity_verified = true
incremental_potential_telescope_verified = true
systematic_resampling_unbiasedness_verified = true
actual_common_target_source_threading_verified = true
finite_resample_move_invariance_verified = true
finite_transition_spectral_gap_verified = true
target_invariance_verified = true

resident_smc_integration_authorized = false
resident_smc_invoked = false
```

Target correctness and finite-state composition do not by themselves certify
finite-particle posterior accuracy. The next admissible phase is a separate
response-free finite-`N`/independent-island error-budget Gate that composes the
registered path and mixing constants with the decision-derived tolerance.
Until it passes, resident execution, predictive calibration, real data,
acquisition, held-out access, confirmation, efficacy, discovery and paper
superiority claims remain blocked.
