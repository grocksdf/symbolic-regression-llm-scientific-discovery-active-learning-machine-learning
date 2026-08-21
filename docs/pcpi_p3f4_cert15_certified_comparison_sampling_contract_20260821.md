# P3F.4-CERT.15 certified comparison and partial-sampling contract

Status: **RESPONSE-FREE DEVELOPMENT ONLY; RESIDENT AND ISLAND EXECUTION REMAIN BLOCKED**

Baseline: `00ecdf01386dc6773e4e5b0f1305210d7430374e`

CERT.14 constructs outward balls for the common collapsed target, bridge
potentials and local/RJ acceptance values.  It deliberately does not turn
those balls into normalized particle weights, ancestor indices or MH
accept/reject outcomes.  CERT.15 defines that missing comparison layer without
reading a response, drawing entropy, running SMC or materializing an island.

The phase is a deterministic correctness proof, not a simulated experiment.
It accepts explicit exact bit strings as fixture inputs and retains an
unresolved comparison as a complete operation failure.  No retry, replacement,
midpoint, favourable endpoint or result-dependent precision increase is
permitted.

## 1. Immutable numerical and bit contract

One plan is bound to the CERT.14 common-target hash and fixes:

- `python-flint==0.8.0` with 512-bit Arb arithmetic;
- 256 unbiased bits per uniform threshold;
- a half-open dyadic-cell interpretation of every bit string;
- monotone outward log normalization;
- multinomial inverse-CDF comparison;
- MH comparison against the certified acceptance-probability ball; and
- `abort-complete-operation-no-retry` for every unresolved comparison.

The 256-bit budget and 512-bit precision are frozen independently of any
target value or observed performance.  CERT.15 has no precision retry and does
not request additional bits after inspecting a comparison.

## 2. Outward log normalization

For certified log-mass balls

\[
  L_i \le \ell_i \le U_i,
  \qquad
  p_i=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}},
\]

the coordinate bounds are evaluated in the overflow-free monotone form

\[
 \underline p_i=
 \left(1+\sum_{j\ne i}e^{U_j-L_i}\right)^{-1},
 \qquad
 \overline p_i=
 \left(1+\sum_{j\ne i}e^{L_j-U_i}\right)^{-1}.
\]

Arb evaluates both expressions outward at the registered precision.  The
exported exact binary endpoints are widened to `[0,1]` when necessary.  The
result must satisfy

\[
 0\le \underline p_i\le\overline p_i\le1,
 \qquad
 \sum_i\underline p_i\le1\le\sum_i\overline p_i.
\]

The final cumulative interval is exactly `[1,1]`, using the proved
normalization identity rather than post-hoc renormalization.  No midpoint or
simplex projection is used.

## 3. Exact-bit uniform thresholds

A 256-bit string with integer value `r` represents the complete set of ideal
uniform reals whose first 256 binary digits match that string:

\[
 I_r=[r2^{-256},(r+1)2^{-256}).
\]

It is not represented by its left endpoint, midpoint or a floating-point
number.  Parsing is a bijection between 32-byte strings and the `2^256`
half-open dyadic cells.  CERT.15 never captures system entropy; the earlier
product-source materialization guards remain false.

This interval-refinement semantics follows the exact random-bit view of the
interval algorithm of Han and Hoshi (IEEE Transactions on Information Theory,
1997).  Fast Loaded Dice Roller gives a complementary exact discrete sampler
for explicitly encoded probabilities, but CERT.15 does not claim that an Arb
probability ball is an exact finite binary probability table:

- https://doi.org/10.1109/18.556109
- https://proceedings.mlr.press/v108/saad20a.html

## 4. Certified multinomial inverse-CDF comparison

Let the exact cumulative boundary after category `k` be `F_k`, with certified
interval `[A_k,B_k]`.  A dyadic threshold cell `I=[u_0,u_1)` is assigned to
category `k` only if

\[
 u_0\ge B_{k-1}
 \quad\text{and}\quad
 u_1\le A_k.
\]

Then every ideal uniform real in that cell and every normalized probability
vector represented by the input balls gives the same inverse-CDF index.  If no
unique category satisfies the inequalities, the comparison is unresolved.

A multinomial batch returns ancestor indices only if every registered
threshold resolves.  One unresolved coordinate raises a complete batch
failure; previously resolved indices are not returned and no replacement bit
string is requested.

For dyadic cell width `delta=2^-b`, the ideal-uniform probability of an
unresolved prefix is bounded by

\[
 \Pr(\mathrm{unresolved})
 \le
 \min\left\{1,
 \sum_{k=1}^{K-1}\big[(B_k-A_k)+2\delta\big]
 \right\}.
\]

This is a union bound over the uncertain internal CDF boundaries.  CERT.15
exports the bound; it does not hide failures or condition a scientific result
on successful draws.  A later integration Gate must compose this probability
with the existing finite-particle decision budget.

## 5. Certified MH uniform comparison

CERT.14 supplies a log-acceptance ball `[a_L,a_U]` with `a_U<=0`.  CERT.15
evaluates

\[
 [q_L,q_U]\supseteq[\exp(a_L),\exp(a_U)]
\]

outward in Arb and compares one dyadic threshold cell `[u_0,u_1)`:

- accept if `u_1 <= q_L`;
- reject if `u_0 >= q_U`;
- otherwise fail unresolved.

The unresolved-probability bound is

\[
 \min\{1,(q_U-q_L)+2\delta\}.
\]

The comparison binds the CERT.14 common-target plan, proposal plan, current
target and proposed target identities.  A crossed target or crossed threshold
purpose fails before a decision is returned.

## 6. Validated numerical semantics

Arb midpoint-radius balls provide mathematical enclosures rather than ordinary
floating estimates.  CERT.15 uses only outward endpoints and exact `Fraction`
arithmetic after export:

- Fredrik Johansson, *Arb: Efficient Arbitrary-Precision Midpoint-Radius
  Interval Arithmetic*, IEEE Transactions on Computers 66(8), 2017,
  https://arxiv.org/abs/1611.02831

The following are forbidden:

- float exponentiation, `softmax`, `logsumexp`, `nextafter` or epsilon padding;
- a probability midpoint, nearest CDF boundary or left-endpoint threshold;
- modulo reduction of random integers;
- alias-table or floating categorical sampling;
- adaptive bit extension, precision retry or favourable endpoint selection;
- retrying an unresolved draw, replacing an island or returning a partial
  ancestor vector.

## 7. Authorization boundary

CERT.15 may authorize only pure response-free comparison composition:

```text
P3F4_CERT15_STANDALONE_COMPARISON_SAMPLING_AUTHORIZED = true
```

All operational switches remain false:

```text
P3F4_CERT15_PRODUCT_BITS_MATERIALIZATION_AUTHORIZED = false
P3F4_CERT15_RESIDENT_RESAMPLING_AUTHORIZED = false
P3F4_CERT15_RESIDENT_MH_DECISION_AUTHORIZED = false
P3F4_CERT15_ISLAND_EXECUTION_AUTHORIZED = false
P3F4_CERT15_RESIDENT_SMC_INTEGRATION_AUTHORIZED = false
P3F4_CERT15_RESIDENT_SMC_RUN_AUTHORIZED = false
```

Every CERT.10--CERT.14 product-source, result-access, island and SMC guard also
remains false.

## 8. Response-free Gate

The deterministic Gate must retain all 124 CERT.3--CERT.14-R1 checks and add
checks for:

1. authorization and complete plan identity;
2. outward monotone normalization and shift invariance;
3. exact probability containment and cumulative endpoints;
4. byte-to-dyadic-cell bijection without modulo reduction;
5. exact finite dyadic inverse-CDF law by enumeration, not simulation;
6. resolved multinomial comparison agreement for every represented value;
7. complete-batch failure on one unresolved comparison;
8. explicit multinomial unresolved-probability bounds;
9. certified MH accept and reject decisions;
10. MH unresolved failure and probability bound;
11. crossed target, purpose and coordinate rejection;
12. absence of floats, midpoints, retry, entropy capture and partial output;
13. retention of every earlier execution guard; and
14. syntax over exactly the Git-tracked Python population.

No response, particle, real dataset, held-out value or simulated outcome is
materialized.

## 9. Remaining integration gap

CERT.15 proves a partial exact comparison kernel whose unresolved event is
explicit.  It does not yet prove that the complete resident Feynman--Kac
program, including all normalization, resampling and MH comparisons, stays
within a frozen joint failure budget.  It also does not connect the external
product-randomness premise to operational bit materialization.

The next admissible phase after a passing CERT.15 Gate is a separate
response-free integration theorem.  It must allocate the comparison-failure
probability jointly with the CERT.9 finite-particle error, bind every bit
coordinate to the CERT.11 product-source manifest, and prove that any
unresolved operation aborts the entire island batch without retry or selective
reporting.  Only after that theorem and its source-composition Gate pass may
resident execution be considered.

Predictive calibration, real data, acquisition, held-out access, confirmation,
efficacy and paper-superiority claims remain blocked.
