# P3F.4-CERT.10 independent-island executor and aggregation contract

Status: **RESPONSE-FREE SOURCE/COMBINATORIAL GATE PASSES; ALL EXECUTION BLOCKED**

GitHub baseline: `880ec013be42304b0efc58098c62753fda08d1d8`

CERT.10 composes the CERT.9 finite-`N` plan with an independent-island executor
interface, exact plan identities, complete failure propagation and the frozen
componentwise-median decision rule. It performs source inspection, exact
rational probability calculations and finite transition/combinatorial checks
only. It does not materialize a production random source or operational class
projector, call `ScalableOpenTargetSMC.run`, access a response, or run a
simulated, formal, real, validation, acquisition, confirmatory or held-out
experiment.

## 1. Randomness boundary: isolation is not independence

The CERT.9 binomial-tail argument requires an actual product law across
islands. Distinct integer values passed to a pseudorandom generator do not, by
themselves, prove that law. NumPy describes spawned streams as independent
with very high probability; that is an engineering guarantee rather than a
mathematical product-law certificate:

- https://numpy.org/doc/2.3/reference/random/parallel.html

Counter-based generators such as Random123 are valuable because a key and
counter give explicit, non-overlapping software coordinates, but deterministic
coordinate separation is still not silently promoted to a proof of physical
or mathematical independence:

- https://random123.com/
- https://www.thesalmons.org/john/random123/papers/random123sc11.pdf

CERT.10 therefore registers an external
`IndependentIslandProductRandomSource` premise. The immutable plan assigns
exactly one ordered coordinate to every island and binds all coordinates to
one `product_law_hash`. The executor rejects:

- a missing, crossed or extra coordinate;
- reuse of one NumPy `Generator` object;
- reuse of one underlying `BitGenerator` state;
- distinct objects carrying an exactly duplicated `BitGenerator` state; and
- any source whose plan or product-law identity differs from the executor.

No `SeedSequence.spawn`, list of integer seeds, or distinct object identity is
claimed to establish independence. CERT.10 proves the finite product-measure
and median-failure combinatorics conditional on the registered product-law
premise. The production implementation of that premise remains unauthorized.

## 2. Exact plan and estimand binding

`ResidentIndependentIslandPlan` binds all of the following in its stable hash:

1. the complete posterior contract;
2. the CERT.8 Feynman--Kac plan;
3. the CERT.9 finite-`N` error-budget plan;
4. every resident particle configuration field;
5. the support-extension-invariant operational-estimand hash;
6. the full-support class-projector hash and ordered class identities;
7. the island count, per-island particle count, coordinate error, MAP budget
   and exact simultaneous-failure fraction; and
8. the product-law and failure-policy identities.

The builder separately revalidates the bridge grid, population relative-ESS
floor, bridge ceiling, resampling law and schedule, terminal population mode,
observation ceiling, class count, particle count, rejuvenation depth, kernel
mixture, MAP budget and simultaneous failure budget. A stable hash alone is
not accepted as a substitute for these runtime equalities.

The old exact-polynomial equivalence aggregation is not the scientific
projector. Every island outcome must be the normalized pushforward of the
frozen full-support operational map `C_star` from the upstream estimand
contract. CERT.10 binds that projector interface and rejects cross-projector
outputs, but deliberately does not implement or authorize the projector.

## 3. Complete failure propagation

The frozen policy is
`collect-all-fail-batch-no-retry-no-replacement`.

Once future execution is separately authorized, the source path must:

1. attempt each registered product coordinate exactly once;
2. retain every coordinate-materialization exception, including exceptions
   whose ordinary string representation is empty;
3. stop before creating an engine if any random coordinate failed;
4. otherwise attempt every island engine and projector exactly once;
5. retain every island index, stream identity, exception type and exception
   message; and
6. return no aggregate if any island failed.

Failure records are checked against the plan and the exact stream-coordinate
hash before `ResidentIndependentIslandBatchFailure` exposes all records. There
is no retry, replacement island, favorable-seed selection, silent omission or
partial aggregation. Setup identity failures also fail closed and cannot be
converted into island results.

## 4. Componentwise medians are decision scores, not probabilities

For class `c`, the registered aggregate is

\[
  m_c=\operatorname{median}
  \{\widehat p_c^{(1)},\ldots,\widehat p_c^{(K)}\}.
\]

Each island vector must be normalized, but the vector of coordinatewise
medians need not lie on the simplex. For example, repeating the three valid
probability vectors

\[
 (0.6,0.4,0,0),\quad(0.4,0,0.6,0),\quad(0,0.6,0.4,0)
\]

across 27 islands gives coordinate medians `(0.4, 0.4, 0.4, 0)`, whose sum is
`1.2`. CERT.10 records the sum and normalization defect. It forbids
normalization, simplex projection and a posterior-probability-vector claim.
Such a repair would change the certified coordinate errors and would be a
post-hoc transformation.

Consequently the CERT.10 aggregate is authorized only for the frozen MAP
decision. Entropy, EIG, posterior-vector, predictive-mixture and calibration
uses remain blocked.

## 5. Direct MAP regret and joint error budget

Suppose every median coordinate obeys `|m_c-p_c| <= epsilon`. If `c_star`
maximizes the true class probabilities and `c_hat` maximizes the median scores,
then

\[
p_{c_\star}-p_{\widehat c}
\le (p_{c_\star}-m_{c_\star})
   +(m_{\widehat c}-p_{\widehat c})
\le 2\varepsilon.
\]

This proof uses only coordinate errors and the argmax inequality; it does not
normalize the median vector or call it a posterior. CERT.9 sets
`epsilon = r_star/C`, so for the required `C >= 2`,
`2 epsilon <= r_star`.

For one coordinate, an odd number `K` of independent islands fails only when a
strict majority of the per-island events fail. The exact registered bound is

\[
B_K=\sum_{j=(K+1)/2}^{K}{K\choose j}(1/4)^j(3/4)^{K-j}.
\]

The simultaneous class statement is the explicit union bound

\[
 C B_K\le\alpha.
\]

The numerator and denominator of this rational value, `C`, `K`, `epsilon`,
`r_star` and `alpha` are all plan-bound. No threshold is selected from an
observed output.

## 6. Actual source composition and hard guard

`ScalableOpenTargetSMC` now accepts either its legacy non-negative integer seed
or one externally supplied NumPy `Generator` plus a nonempty stream identity.
Combining the two modes is rejected. The externally supplied object is used
directly and its identity is carried into the result evidence record.

The island executor injects exactly the generator belonging to the current
product coordinate, then validates the projector output and aggregate
identity. Three independent authorization flags remain false. The top-level
`run()` guard precedes coordinate creation, product-source access, response or
projector access, engine construction and SMC execution. Thus the unreachable
body proves source composition; it is not evidence from executing an island.

## 7. Response-free Gate

CERT.10 retains all 56 CERT.3--CERT.9 checks and adds ten checks covering:

1. exact Feynman--Kac, finite-`N`, particle, estimand and claim binding;
2. rejection of cross-target, bridge-grid, configuration and class identities;
3. unique product coordinates and exact finite product measures;
4. rejection of aliased or exactly duplicated generator state;
5. normalized, identity-bound per-island operational pushforwards;
6. the non-simplex componentwise-median counterexample and claim boundary;
7. exact binomial-tail/union error budgeting;
8. rejection of missing, duplicate and cross-plan island outcomes;
9. complete, coordinate-bound failure propagation without partial output; and
10. the actual source order, external-generator injection and hard guard.

The registered identity is `66/66`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

## 8. Authorization boundary

CERT.10 establishes:

```text
independent_product_randomness_contract_verified = true
random_stream_alias_rejected = true
exact_plan_identity_binding_verified = true
operational_estimand_binding_verified = true
all_island_failures_propagated = true
componentwise_median_score_aggregation_verified = true
simultaneous_union_error_budget_verified = true
map_decision_regret_certificate_verified = true

distinct_integer_seeds_treated_as_independent = false
retry_or_island_replacement_authorized = false
partial_island_aggregation_authorized = false
componentwise_median_normalization_assumed = false
posterior_probability_vector_claimed = false
independent_product_random_source_implementation_authorized = false
operational_class_projector_implementation_authorized = false
independent_island_execution_authorized = false
resident_smc_integration_authorized = false
resident_smc_invoked = false
```

The next admissible phase is a separate response-free Gate for an auditable
product-random-source implementation and the frozen full-support operational
class projector. It must preserve the explicit idealized-randomness premise,
process/state isolation, exact projector identity and boundary-uncertainty
semantics. Until both implementations pass, no CERT.10 execution flag may be
changed and island/resident SMC, predictive calibration, real data,
acquisition, held-out access, confirmation, efficacy, discovery and paper
superiority claims remain blocked.
