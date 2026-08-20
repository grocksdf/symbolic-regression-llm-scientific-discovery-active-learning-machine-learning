# P3F.4-CERT.11 product source and full-support projector contract

Status: **RESPONSE-FREE SOURCE GATE PASSES; ALL EXECUTION REMAINS BLOCKED**

GitHub baseline: `9f8ca9caf1ca863d05cb22e92272a3a5f533ecad`

CERT.11 implements the two source boundaries left open by CERT.10: an
auditable direct-key product-random-source and the frozen grid-restricted
operational class map on its full implicit range. The Gate uses source
inspection, exact rational arithmetic and complete finite combinatorial checks
only. It does not capture operating-system entropy, instantiate a production
random stream, access a particle result, evaluate a numerical predictive CDF,
call an island executor or run resident SMC. No simulated, formal, real,
validation, acquisition, confirmatory or held-out experiment is run.

## 1. Product randomness claim boundary

Let the CERT.10 plan register ordered island coordinates

\[
  \xi_0,\ldots,\xi_{K-1}
\]

and the product-law identity `product_law_hash`. CERT.11 binds one 128-bit key
`k_i` directly to every `xi_i`. If future authorization is granted, the
corresponding generator construction is exactly

```python
np.random.Generator(np.random.Philox(key=k_i, counter=0))
```

NumPy's Philox documentation states that a direct key lies in
`[0, 2**128)`, the counter lies in `[0, 2**256)`, and supplying `key`
bypasses `SeedSequence`:

- https://numpy.org/doc/2.3/reference/random/bit_generators/philox.html

Python documents `secrets.token_bytes` as drawing bytes from the strongest
operating-system randomness source provided by the platform:

- https://docs.python.org/3/library/secrets.html

Those engineering interfaces do not prove that the physical or operating-
system key tuple has a mathematical product distribution. CERT.11 therefore
retains the exact external premise
`external-independent-os-entropy-key-tuple` and reports

```text
external_independence_premise_proved_by_source = false
```

The finite product-law and independent-island median theorems remain
conditional on that premise. Distinct integer seeds, distinct Python objects,
`SeedSequence.spawn` children and jumped streams are not substituted for it.

## 2. Ordered key manifest and failure semantics

`ResidentPhiloxProductSourceContract` hashes the island-plan identity,
product-law identity, exact ordered coordinate hashes, Philox algorithm,
128-bit direct-key construction and zero initial counter.

`ResidentPhiloxKeyManifest` then binds:

1. exactly one 16-byte key to each registered coordinate;
2. the exact coordinate order and source-contract hash;
3. one entropy capture per coordinate;
4. zero retries and zero replacement coordinates; and
5. a commitment to every key in the public audit record without printing the
   raw key bytes in `repr` or `audit_record`.

Future entropy capture contains exactly one `secrets.token_bytes(16)` call per
coordinate. There is no root seed or root key, child derivation, spawn, jump,
`while` loop, collision regeneration or observed-output selection. A duplicate
key makes manifest construction fail. It is not silently repaired, because a
repair loop would change the preregistered one-capture law and introduce an
unrecorded selection event.

Each coordinate is one-shot. Reuse fails rather than advancing, resetting or
replacing its generator. Cross-plan, cross-product-law, missing, extra and
reordered coordinates also fail closed. The existing CERT.10 batch policy
continues to preserve every island failure and forbids partial aggregation.

## 3. Frozen operational estimand

CERT.11 instantiates the upstream G0 contract. Before future acquisition it
freezes:

- the initial-history hash `H0`;
- the initial center/scale standardizer hash;
- a sorted, unique, selection-visible action grid `A0`;
- seven strictly ordered response thresholds derived from the frozen
  probability levels

  \[
    (0.05,0.15,0.30,0.50,0.70,0.85,0.95);
  \]

- future budget `B=32`; and
- `K_B=ceil(sqrt(B))=6` with left-closed, right-open bins and a closed final
  endpoint.

For every state `z` in the countably open support, including a state never
seen in a particle population, define

\[
  \Psi_0(z)
  =\big(F_z(r_j\mid a_i,H_0)\big)_{i,j}
\]

and

\[
  q_B(u)=\min\{K_B-1,\lfloor K_Bu\rfloor\},\qquad
  C_\star(z)=\big(q_B(\Psi_{0,k}(z))\big)_k.
\]

The implementation does not inspect an enumerated structure bank, particle
order, discovered-class set or exact-polynomial identifier to define this
label. Adding or reordering supported states therefore cannot relabel an
existing state. The scientific claim remains restricted to the registered
action/threshold grid; it is not a continuum predictive-equivalence claim.

## 4. Complete implicit class space

If there are `d=7*|A0|` CDF coordinates, the theoretical class space is

\[
  \{0,\ldots,5\}^{d},\qquad C=6^d.
\]

CERT.11 does not enumerate this potentially enormous Cartesian product. For a
signature `b=(b_0,...,b_{d-1})`, it uses the reversible base-six rank

\[
  r(b)=\sum_{j=0}^{d-1} b_j6^{d-1-j},
  \qquad 0\le r<6^d.
\]

The class ID contains the complete estimand hash and this rank. Rank/unrank is
checked on every one of the `6^7=279936` classes in the smallest registered
one-action space. The mapping is consequently defined on the complete class
space without making observed classes determine the label system.

No `other` class, hash bucket, nearest centroid, merge order or result-derived
compression is introduced.

## 5. Certified boundary uncertainty

The mathematical CDF determines `C_star`; an approximate floating CDF does
not. CERT.11 accepts only an exact rational outward enclosure `[l_k,u_k]` from
a separately identified certified interval oracle.

For coordinate `k`, the possible bin set is exactly

\[
  Q_k=\{q_B(u):u\in[l_k,u_k]\}.
\]

If every `Q_k` is a singleton, the state's class is exact. Otherwise the state
is recorded as `boundary-uncertain` with its exact posterior mass and the
tuple of possible bin ranges. The state is never assigned to its nearest bin.

For any queried full class signature `c`, the sparse projection reports

\[
 L_c=m_c^{\mathrm{exact}},\qquad
 U_c=m_c^{\mathrm{exact}}
     +\sum_{z:\ c\in Q(z)}m_z.
\]

This produces rigorous marginal class-mass bounds without enumerating all
combinations in `Q(z)` or all `6^d` classes. Exact state masses must already
sum to one. There is no renormalization, simplex projection, clipping or
regularizer.

## 6. Fixed-vector CERT.10 adapter fails closed

The CERT.10 executor interface returns a vector aligned to its preregistered
`class_ids`. CERT.11 permits that adapter only when:

1. every particle has a singleton certified class;
2. every occupied full-support class is in the preregistered list; and
3. the unchanged empirical weights already sum to one within the registered
   floating identity tolerance.

Any boundary-uncertain mass or occupied but unregistered class raises an
error. The adapter neither appends an observed `other` class nor silently
renormalizes a partial vector. Thus implementing the full map does not pretend
that CERT.10's current finite class list exhausts `6^d`.

## 7. Rigorous numerical-oracle boundary

`OpenTargetParticleSnapshot.predictive_cdf` uses the ordinary SciPy Student-t
CDF and returns one floating number. That is useful numerically, but it is not
an outward interval proof. Wrapping it with `nextafter`, adding an empirical
tolerance or choosing the nearest class bin would not establish containment of
the mathematical CDF.

CERT.11 therefore defines `CertifiedPredictiveCDFIntervalOracle` as an
explicit protocol and binds its contract hash, initial history, estimand,
full-open support and no-future-response properties. It does not claim to
implement that oracle. A future implementation may use a rigorously verified
ball-arithmetic incomplete-beta calculation; the FLINT/Arb documentation
describes rigorous real/complex ball enclosures and hypergeometric functions:

- https://flintlib.org/doc/index_arb.html
- https://flintlib.org/doc/acb_hypgeom.html

No such dependency is silently added in CERT.11. Consequently

```text
certified_cdf_interval_oracle_implementation_authorized = false
projector_result_access_authorized = false
```

## 8. Source guards and unchanged CERT.10 authorization

Four new guards are false:

```text
P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED = false
P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED = false
P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED = false
P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED = false
```

The three CERT.10 guards remain literally unchanged and false:

```text
P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED = false
P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED = false
P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED = false
```

Entropy capture is guarded before plan access. Stream materialization is
guarded before coordinate access. Projector execution is guarded before
result, particle or interval-oracle access. Therefore the implementation can
be audited without drawing randomness or exposing a response-dependent path.

## 9. Response-free Gate

CERT.11 retains all 66 CERT.3--CERT.10 checks and adds 13 checks covering:

1. all old and new authorization guards;
2. exact ordered product-coordinate binding;
3. manifest one-to-one key range, commitments and failure semantics;
4. guard ordering and direct `Philox(key=..., counter=0)` source composition;
5. frozen initial history, grids, `B=32`, `K_B=6` and claim boundary;
6. the complete small-space base-six rank/unrank bijection;
7. support-extension and population-order invariance;
8. exact endpoint and cross-boundary bin semantics;
9. sparse exact pushforward mass conservation;
10. exact lower/upper propagation of boundary-uncertain mass;
11. fixed-vector rejection of uncertainty and unregistered occupied classes;
12. actual projector plan binding and guard ordering; and
13. absence of ordinary floating-CDF certification, nearest-bin assignment and
    Cartesian class enumeration.

The registered identity is `79/79`. Every repository Python file outside
`.git`, `.venv` and `evidence` is syntax checked.

## 10. Authorization result and next admissible phase

CERT.11 establishes the source and exact combinatorial implementation of the
product-coordinate binding and full-support operational map. It does not
establish that an actual OS key tuple realizes an ideal mathematical product
law, and it does not provide a rigorous Student-t CDF interval implementation.
It also does not reconcile the full implicit class count `6^d` with the
CERT.9 finite-`N` simultaneous union budget and CERT.10 fixed-vector plan.

The next admissible phase is a separate response-free Gate that:

1. implements and independently verifies a rigorous predictive Student-t CDF
   interval oracle for every full-open resident state;
2. binds its dependency version and precision/refinement termination contract;
3. reconciles the complete implicit class space with the finite-`N` joint
   decision-error budget without observed-class selection or post-hoc
   normalization; and
4. proves that the actual CERT.10 projector adapter cannot omit class mass.

Until that Gate passes, island/resident SMC, entropy capture, stream
materialization, projector result access, predictive calibration, real data,
acquisition, held-out access, confirmation, efficacy, discovery and paper
superiority claims remain blocked.
