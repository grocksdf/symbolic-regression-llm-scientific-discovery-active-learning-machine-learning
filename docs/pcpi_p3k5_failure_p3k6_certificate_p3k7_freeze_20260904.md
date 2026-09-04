# P3K.5 terminal audit, P3K.6 finite certificate, and P3K.7 freeze

## Preserved P3K.5 result

The source-frozen P3K.5 run stopped after 10,163.3 seconds while starting CCPP
seed 2026080707's PCPI policy run. At query 27 the P3K.4 projection produced a
nonempty singleton admissible set, so it had repaired P3K.3's structural
empty-set failure. The decision transaction then rejected an infinite scalar
while constructing strict JSON. No query-27 decision or response was published,
validation was not used for selection, and held-out remained closed. The output
`outputs/p3k5_projected_guard_real_acquisition_a4bebe03_20260904` is terminal
and cannot be resumed, overwritten, or relabelled as efficacy evidence.

## Root cause

The lower-envelope ranking code historically represented a certified ranking
over one eligible candidate by

\[
  (\text{margin},\text{error},\text{gap})=(+\infty,0,+\infty).
\]

This is a mathematically suggestive convention, but it is not an admissible
number in the strict, portable JSON evidence protocol. The serializer uses
`allow_nan=False` and therefore failed closed. This was neither a nonfinite
acquisition score nor a failure of the shared-innovation posterior.

## P3K.6 canonical singleton certificate

Let the projected admissible candidate set be \(A\). If \(|A|=1\), its sole
element is the maximizer over \(A\) without any pairwise comparison. P3K.6
encodes that proposition as

\[
  (\text{margin},\text{error},\text{gap})=(0,0,0),\qquad
  \text{certified}=\mathrm{true},qquad
  |\text{possible maximizers}|=1,
\]

under the distinct method identity
`singleton-projected-admissible-set-vacuous-rank-certificate-v1`. The singleton
admissible mask remains part of the decision record. Thus zero does not assert
a positive pairwise gap; the method identity says that no competitor exists in
the certificate domain.

Both the base quadrature certificate and the finite-model lower-envelope
certificate now use finite neutral values for singleton domains. The P3K
class-conditional finalizer binds the distinct singleton method, the exact
admissible mask, and possible-maximizer count one. The no-data P3K.6 Gate
exercises the complete four-power class-conditional scorer, selects the sole
admissible candidate, serializes the entire decision under strict JSON, and
round-trips it without response access. No random, simulated, real,
validation-response, or held-out surface is used.

## P3K.7 formal boundary

P3K.7 changes only singleton certificate representation and audit relative to
P3K.5. It freezes the same three provenance-verified real datasets, eight seeds,
32 initial observations split into 16 base and 16 residual observations, 32
acquisitions, 128-candidate pool, 256 validation observations, four matched
policies, four likelihood powers, quadrature schedule, P3K shared-innovation
state, minimum-violation representative projection, assessment rules, runtime,
fail-fast policy, and closed held-out boundary.

Every singleton projected query must use the new method with finite zero
certificate scalars and exactly one possible maximizer; every nonsingleton query
must not use it. The method is recorded in the manifest. P3K.7 is still
failure-informed real-development evidence, not independent confirmation.
