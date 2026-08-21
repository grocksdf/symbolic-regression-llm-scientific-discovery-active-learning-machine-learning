# P3F.4-CERT.16 joint failure-budget and product-bit contract

Status: **RESPONSE-FREE CONDITIONAL INTEGRATION THEOREM; ALL EXECUTION BLOCKED**

Baseline: `7f8f961900c8f1cd739bb84d23c33c57f848640e`

CERT.16 composes the CERT.9 finite-particle decision error, the CERT.10
independent-island batch, the CERT.11 ordered key manifest, the CERT.14 common
target and the CERT.15 certified comparison layer. It does not capture entropy,
materialize a random bit, inspect a response, create a particle, run an island
or return a scientific result.

## 1. Joint failure event

Let `alpha` be the frozen simultaneous-failure probability in the CERT.9 and
CERT.10 plans and let `B_N` be CERT.9's exact independent-island median bound.
The numerical-comparison allowance is not a new tolerance:

\[
  \Delta = \alpha-B_N.
\]

The integration plan is invalid unless `Delta` is strictly positive. It may
not increase `alpha`, reduce the operational class count, change the island
count or weaken the finite-`N` theorem to manufacture slack.

For frozen bounds

```text
K = independent island count
N = particles per island
S = maximum observations * maximum bridges per observation
M = maximum rejuvenation steps per bridge
```

the maximum comparison counts are

\[
 Q_R=KNS,
 \qquad
 Q_M=KNSM,
 \qquad
 Q=Q_R+Q_M.
\]

`Q_R` covers one multinomial inverse-CDF threshold per offspring at every
bridge. `Q_M` covers one MH accept/reject threshold per particle and registered
rejuvenation step. The common per-comparison allowance is frozen before any
target value is available:

\[
 \rho=\Delta/Q.
\]

Before a future execution may read or materialize the threshold bits for a
coordinate, it must check that CERT.15's unresolved-probability upper bound is
at most `rho`. If the check fails, the entire independent-island batch aborts
before the threshold is observed. A threshold that would happen to resolve
cannot rescue an over-budget comparison.

This runtime check is necessary for failure closure but is not itself a
probability theorem. If a reachable pre-threshold state has a bound above
`rho`, deterministic rejection in that state has conditional probability one,
not the numerical unresolved bound. Therefore CERT.16 does **not** claim a
bounded completion probability from runtime checks.

Only conditional on a separate response-free theorem proving, uniformly over
every reachable pre-threshold state and every coordinate, that the unresolved
bound is at most `rho`, does the union bound give

\[
 \Pr(\text{wrong MAP decision or comparison abort})
 \le B_N+Q\rho
 =B_N+\Delta
 =\alpha.
\]

No independence between comparison-failure events is claimed or required for
this conditional union bound. CERT.16 freezes and verifies the algebraic budget
identity, while `uniform_reachable_state_comparison_bound_verified` remains
false.

## 2. Implicit complete coordinate space

The comparison space is represented by an exact rank bijection rather than a
materialized list. The lexicographic order is:

1. island index;
2. path-step index;
3. all `N` multinomial offspring coordinates; then
4. for each rejuvenation step, all `N` MH coordinates.

Every coordinate binds:

- the CERT.16 plan hash;
- the CERT.15 sampling-plan hash and purpose;
- the CERT.10 island plan and exact island-stream coordinate hash;
- the CERT.11 product-source contract and key-manifest hash;
- the key commitment for that island, never the raw key;
- island, path-step, particle and optional rejuvenation-step indices; and
- one domain-separated Philox counter address.

Rank/unrank must be a bijection on `[0,Q)`. Missing, duplicate, out-of-range or
cross-plan coordinates fail closed.

## 3. Philox counter-domain separation

One comparison threshold uses exactly one 256-bit Philox block, represented as
four 64-bit words in generator-output order and encoded big-endian per word.
The counter address is

\[
 c=(d\ll192)+r_i,
\]

where `d = 0x5043504943455254` (ASCII `PCPICERT`) is the fixed 64-bit comparison
domain and `r_i` is the within-island comparison rank. CERT.16 requires
`r_i < 2^192`. The high domain separates comparison blocks from the ordinary
low-counter resident stream; each island uses its own CERT.11 key commitment.

This proves deterministic address uniqueness and prevents address reuse. It does
not prove that Philox output is a mathematical product of unbiased physical
bits. The exact-bit theorem is therefore conditional on an external ideal-bit
premise, and product-bit materialization remains unauthorized.

## 4. Failure closure

The policy is

```text
precheck-bound-then-read-one-coordinate-abort-entire-batch-no-retry
```

For every coordinate:

1. validate its exact rank, target, stream and manifest identity;
2. compute the CERT.15 unresolved-probability upper bound;
3. reject the complete batch if the bound exceeds `rho`;
4. only then, in a future separately authorized source, read its one-shot bit
   block and perform the certified comparison; and
5. abort the complete batch on an unresolved comparison.

No extra bits, precision retry, replacement key, replacement island, partial
ancestor vector, partial island aggregate, successful-island subset or
favourable-coordinate reporting is permitted. Failure identity includes the
exact coordinate rank and hash. The deterministic lexicographic order prevents
choosing which failure to expose.

## 5. Randomness claim boundary

CERT.11 directly binds one 128-bit key to each ordered island coordinate and
rejects key collision without retry. CERT.16 refines each island coordinate by
an injective counter address. These are auditable software identities.

Neither source inspection nor distinct Philox keys prove mathematical
independence or ideal unbiased bits. CERT.16 explicitly records:

```text
philox_address_uniqueness_verified = true
philox_pseudorandomness_promoted_to_mathematical_independence = false
external_ideal_bit_product_law_required = true
external_ideal_bit_product_law_implementation_authorized = false
uniform_reachable_state_comparison_bound_verified = false
```

This boundary cannot be weakened by calling distinct seeds, keys, generators
or counters independent evidence.

## 6. Authorization boundary

CERT.16 may authorize only construction and finite response-free auditing of
the integration theorem:

```text
P3F4_CERT16_STANDALONE_INTEGRATION_THEOREM_AUTHORIZED = true
```

Every operational flag remains false:

```text
P3F4_CERT16_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = false
P3F4_CERT16_RESIDENT_COMPARISON_INTEGRATION_AUTHORIZED = false
P3F4_CERT16_ISLAND_BATCH_EXECUTION_AUTHORIZED = false
P3F4_CERT16_RESIDENT_SMC_RUN_AUTHORIZED = false
P3F4_CERT16_EXTERNAL_IDEAL_BIT_PRODUCT_LAW_IMPLEMENTATION_AUTHORIZED = false
```

All CERT.9--CERT.15 execution, entropy, result-access, projector and island
guards remain false.

## 7. Response-free Gate

The Gate must retain all 138 CERT.3--CERT.15 checks and add tests for:

1. authorization and complete cross-phase identity binding;
2. exact residual budget and conditional `B_N + Q rho = alpha` identity;
3. exact resampling, MH and total coordinate counts;
4. rank/unrank bijection on a complete small space and endpoints of the full
   registered space;
5. binding to every CERT.11 island coordinate and key commitment;
6. injective high-domain Philox counter addresses;
7. purpose and rejuvenation-index consistency;
8. rejection of crossed common target, finite-N, manifest and source plans;
9. pre-bit rejection of an unresolved bound above `rho`;
10. acceptance of a local bound at or below `rho` without claiming a uniform
    reachable-state envelope or completion probability;
11. whole-batch failure identity with no partial output or retry;
12. source ordering: guard, bound check, then bit materialization;
13. retention of every earlier operational guard; and
14. syntax over the exact Git-tracked Python population.

No experiment or random draw is part of this Gate.

## 8. Remaining blocker

A passing CERT.16 Gate proves the conditional joint-budget identity and the
complete software address space. It still cannot authorize resident execution
for two independent reasons:

1. no response-free uniform theorem yet bounds CERT.15 comparison ambiguity by
   `rho` over every reachable state; and
2. the CERT.11 Philox implementation supplies an auditable PRNG construction,
   not a proof of mathematical ideal-bit independence.

The next phase must address both premises or reformulate the scientific theorem
so its completion and computational-randomness assumptions are stated and
audited accurately. Real data, held-out access, acquisition, confirmation,
efficacy and paper-superiority claims remain blocked.
