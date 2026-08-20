# P3F.4-CERT.9 finite-N and independent-island error-budget contract

Status: **RESPONSE-FREE THEOREM/SOURCE GATE PASSES; RESIDENT EXECUTION BLOCKED**

GitHub baseline: `ae84761b0ec02b9429810b0f85489d4993755015`

CERT.9 converts the CERT.8 common-target path into a nonasymptotic error-budget
contract for bounded operational-class probabilities. It performs only source,
algebraic, exact-rational and finite-state combinatorial checks. It does not call
`ScalableOpenTargetSMC.run`, create an island executor, materialize a response,
or run simulated, formal, real, validation, acquisition, confirmatory or
held-out experiments.

## 1. Two theorem blockers and the source-level repair

### 1.1 Systematic offspring are not a conditional product law

CERT.8 proves exact unbiasedness of randomized systematic resampling. That fact
is insufficient for the finite-sample theorem used here. The fixed-path theorem
of Marion, Mathews and Schmidler draws every offspring conditionally
independently from the weighted empirical population. Systematic resampling uses
one shared offset. For two weights `(1/2, 1/2)` and two offspring, its exact law
is concentrated on `(0, 1)`, whereas multinomial resampling assigns probability
`1/4` to all four ordered outcomes. CERT.9 therefore rejects any transfer of the
theorem to the systematic branch.

This is not a variance preference chosen after observing a result. Order-
dependent failures of systematic-resampling convergence are documented by
Gerber, Chopin and Whiteley, and systematic resampling is not uniformly
variance-dominating in the counterexample of Douc, Cappé and Moulines:

- https://arxiv.org/abs/1707.01845
- https://arxiv.org/abs/cs/0507025

The finite-`N` resident identity uses conditionally independent multinomial
offspring after every bridge. The historical CERT.8 systematic path remains
preserved as an unbiased common-target composition result.

### 1.2 A finite censored spectral gap is not a full-open mixing constant

CERT.8 reconstructed a finite censored local/RJ transition matrix and proved a
positive finite spectral gap. A countably-open raw-AST state space has no finite
maximum address count, so that finite result cannot be extrapolated into a
uniform production spectral gap.

CERT.9 uses a random-scan kernel mixture instead:

\[
K_s=\eta K_s^{\mathrm{prior}}+(1-\eta)K_s^{\mathrm{local/RJ}},
\qquad 0<\eta\le 1.
\]

Both component kernels leave the same CERT.8 bridge target invariant. If the
collapsed likelihood is bounded by the response-energy envelope `M_s` and the
semantic core supplies `Z_{J,s} <= Z_s`, the exact-prior proposal satisfies

\[
p_0(z)\ge \frac{Z_s}{M_s}\pi_s(z)
          \ge \frac{Z_{J,s}}{M_s}\pi_s(z).
\]

The independence-MH kernel and its random-scan mixture therefore obey

\[
K_s^{\mathrm{prior}}(z,\cdot)
 \ge \epsilon_s\pi_s(\cdot),\qquad
K_s(z,\cdot)\ge \eta\epsilon_s\pi_s(\cdot),
\quad
\epsilon_s=Z_{J,s}/M_s.
\]

Consequently, from any starting law,

\[
\|\nu K_s^m-\pi_s\|_{\rm TV}
\le(1-\eta\epsilon_s)^m.
\]

This retains local/RJ exploration while the exact-prior component supplies the
global countably-open guarantee. The mixture probability is part of the stable
plan identity and cannot adapt to responses or empirical acceptance.

The resident particle implementation also carries the raw/component prior in
log space, computed from the exact arbitrary-precision prior mass. The legacy
floating probability is retained only for finite-support diagnostics. A valid
301-node response-free witness underflows that legacy field to zero while its
log prior remains finite, and the actual prior-independence/local-RJ acceptance
path uses only the finite log mass. Consequently, floating underflow cannot
delete a legal open-support state or invalidate the minorization premise.

## 2. Fixed-path finite-sample budget

The primary theorem is Theorem 1 in the 26 August 2025 revision of:

- Marion, Mathews and Schmidler, *Finite Sample Bounds for Sequential Monte
  Carlo and Adaptive Path Selection Using the L2 Norm*:
  https://arxiv.org/pdf/1807.01346

For at most `S` bridge targets and adjacent population relative ESS at least
`E`,

\[
\left\|\frac{\pi_s}{\pi_{s-1}}\right\|_{L_2(\pi_{s-1})}^2
\le E^{-1}.
\]

For any bounded functional `|f| <= 1`, the registered sufficient particle
count is

\[
N\ge
\log(128S)\max\left\{\frac{18}{E},\frac{1}{2\varepsilon^2}\right\}.
\]

The required per-bridge kernel accuracy is

\[
a=(8NS)^{-1}.
\]

CERT.9 computes the smallest integer `N` above the theorem expression and, for
each already-certified bridge, the smallest integer

\[
m_s=\left\lceil
\frac{\log a}{\log(1-\eta\epsilon_s)}
\right\rceil.
\]

Every response-prefix bridge and its mixing depth are certified before the
resident source may create its first particle. If the analytic path is absent,
the observation bound is exceeded, or any `m_s` exceeds the frozen ceiling,
the path fails closed. Empirical CESS, acceptance and genealogy cannot repair a
failed theorem premise.

The source uses the conservative bound

\[
S=(\text{maximum observations})
  (\text{maximum bridges per observation}).
\]

The observation bound, class count, regret budget, failure probability, mixture
probability and rejuvenation ceiling must be supplied explicitly for the
finite-`N` mode. They are not inferred from an SMC result.

## 3. From bounded class coordinates to scientific decision error

Let the operational class count be `C`, and let `r_*` be the frozen 0--1 MAP
regret budget. CERT.9 registers every class indicator as one bounded
functional and sets

\[
\varepsilon=r_*/C.
\]

If all class-coordinate estimates are within `epsilon`, then

\[
\|\widehat p-p\|_{\rm TV}
\le \frac{C\varepsilon}{2}=\frac{r_*}{2},
\]

and the plug-in MAP decision has posterior 0--1 regret at most `r_*`. This is a
decision-derived accuracy budget. A numerical tolerance is not selected from a
particle result, and no favorable class coordinate is singled out.

The finite fixture in the static Gate uses `C=4`, `r_*=0.02`, five observation
slots and the already registered 64-step bridge ceiling only to verify the
formula and source binding. These fixture values are not a production freeze.

## 4. Independent-island confidence amplification

The finite-sample theorem gives one island success probability at least `3/4`
for one registered bounded functional. Particles within that island remain an
interacting particle system and are not treated as independent replicates.

For an odd number `K` of genuinely independent islands, the componentwise
median fails for one functional only if at least `(K+1)/2` islands fail. The
worst-case probability is evaluated exactly:

\[
B_K=\sum_{j=(K+1)/2}^{K}
{K\choose j}(1/4)^j(3/4)^{K-j}.
\]

CERT.9 chooses the smallest odd `K` satisfying

\[
C B_K\le\alpha,
\]

where `alpha` is the frozen simultaneous failure probability. The factor `C`
is an explicit union bound over every class coordinate. No individual island,
seed, particle or favorable mean can replace the componentwise median.

This separation follows the replication discipline in the upstream estimand
contract. The island-particle literature explains why within-island population
size and island count are distinct approximation axes:

- Vergé, Dubarry, Del Moral and Moulines, *On parallel implementation of
  sequential Monte Carlo methods: the island particle model*:
  https://arxiv.org/pdf/1306.3911

CERT.9 proves the product-law and exact binomial combinatorics but deliberately
does not create an island executor or claim that distinct integer seeds alone
constitute mathematical independence.

## 5. Actual resident source composition

The finite-`N` resident source is accepted only when all of the following are
part of the immutable configuration:

```text
maximum_nodes = None
tempering_mode = certified-population-relative-ess
resampling_kind = multinomial
resampling_schedule = post-bridge-always
rejuvenation_population_mode = terminal-only
proposal_kind = raw-state-local-rj
0 < proposal_mixture_weight <= 1
particle_count = theorem-derived N
rejuvenation_steps = frozen analytic ceiling
```

Before any particle draw, the actual `run()` source enumerates the complete
response-prefix bridge path and constructs every mixing budget. During the
unreachable execution body, the live bridge must equal that preflight identity,
multinomial resampling is revalidated at the same target, and `_rejuvenate`
requires the matching bridge budget before choosing either the prior-
independence or local/RJ component.

The top-level authorization flag is still false before data validation,
preflight and particle sampling. Thus these are source-composition guarantees,
not evidence from an SMC run.

## 6. Response-free Gate

CERT.9 retains all 46 CERT.3--CERT.8 checks and adds ten checks covering:

1. exact fixed-path theorem identity and rejection of systematic transfer;
2. theorem-derived particle count and class decision budget;
3. exact independent-island median/union failure budget;
4. multinomial conditional product law versus the systematic shared offset;
5. exact prior-independence/local-RJ mixture invariance and minorization;
6. an analytic mixing budget for every response-free fixture bridge;
7. fail-closed insufficient mixing and cross-target bindings;
8. exact runtime counts and resampling/kernel controls;
9. actual source preflight, every-bridge resampling, kernel mixture and
   underflow-safe open prior log mass; and
10. the execution guard before data, preflight and particle sampling.

The registered identity is `56/56`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

## 7. Authorization boundary

CERT.9 establishes:

```text
finite_n_l2_particle_bound_verified = true
decision_regret_budget_verified = true
multinomial_product_resampling_verified = true
systematic_finite_n_theorem_transfer_rejected = true
countably_open_kernel_minorization_verified = true
prior_local_kernel_mixture_invariance_verified = true
finite_n_bridge_preflight_verified = true
independent_island_product_law_verified = true
independent_island_median_budget_verified = true
within_island_particle_independence_assumed = false

independent_island_execution_authorized = false
resident_smc_integration_authorized = false
resident_smc_invoked = false
```

The next admissible phase is a separate response-free independent-island
executor and aggregation source Gate. It must bind exactly one CERT.9 plan,
create isolated random streams and engines, preserve every island failure, and
return only the registered componentwise medians and simultaneous certificate.
Until that Gate passes, resident execution, predictive calibration, real data,
acquisition, held-out access, confirmation, efficacy, discovery and paper
superiority claims remain blocked.
