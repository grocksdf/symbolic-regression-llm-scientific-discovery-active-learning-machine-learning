# PCPI P3I.3 immutable protocol-failure audit

## Terminal status

The unique user-executed P3I.3 run is complete but protocol-invalid. It must
not be resumed, overwritten or counted as efficacy evidence. The preserved
output is
`outputs/p3i3_decision_targeted_real_acquisition_3a496dbc_20260831`.

The run is bound to source commit
`3a496dbce44c038240409a1efb1eb771283ec829`, tree
`11d56faa466bb359bbac27e3870aac200a2af352` and config SHA-256
`2f92c97ddf6099a1e39f0aa09c9edeed69da662b388bec37a11a207964281fd5`.
All 96 policy runs and 3072 queries completed with zero execution failures.
The evidence registry contains 97 events and verifies successfully. Official
data hashes, source identity, dependency hash and split commitments verify;
held-out remained closed and selection never used held-out responses.

Nevertheless, `protocol_gate_passed` is false and the terminal assessment is
`INVALID_PROTOCOL_FAILURE`. Consequently both formal protocol and efficacy
evidence flags are false, as required.

## Unique failed decision and diagnosis

Exactly one registered decision failed:
`initial_class_partition_shared_across_policies=false`. In all 24
dataset/seed groups, random, uncertainty and QBC shared one initial partition
and common-scale hash, while PCPI had a second hash. All 24 groups retained the
same operational class count, and the largest absolute difference in initial
frozen-class entropy was only `3.02e-14`.

The discrepancy is structural and deterministic, not an efficacy-dependent
signal. Matched baselines reconstructed the eta-one H0 posterior once with
complete-history batch sufficient statistics. PCPI obtained the same
mathematical conjugate target through the P3H path: 16 batch warmup updates
followed by 16 strict-prefix sequential updates needed to train the residual
law. Matrix multiplication and pointwise accumulation differ in floating-point
summation order. P3I.1 deliberately included the raw common-scale byte hash in
each decision-class identity; therefore last-bit scale differences created
different class IDs even when class counts and entropies agreed.

The Gate correctly rejected the mismatch. Weakening the Gate, rounding the
scale hash, copying PCPI's hash into baseline records or accepting approximate
identity would conceal that the preregistered claim required one actual frozen
target shared by all policies.

## Supervisor diagnostic defect

The Python process correctly returned exit code 2 after writing its complete
invalid-protocol terminal artifacts. Windows PowerShell 5.1 returned a null
`Process.ExitCode` when `Start-Process -PassThru` was combined with
`-RedirectStandardOutput` and `-RedirectStandardError`, so the supervisor's
error message omitted the numeral. This display defect did not change the
runner result or evidence. P3I.4 avoids the affected API composition and tests
that a nonzero encoded-child exit is preserved.

## Admissible continuation

P3I.4 may repair only source identity and execution supervision:

1. Construct one canonical eta-one H0 posterior from the complete already-open
   initial history per dataset and seed.
2. Freeze one decision-regret class target from that posterior before entering
   the policy loop and inject the same immutable object into every policy.
3. Retain the P3H strict-prefix residual history and its four power-specific
   posterior/residual states for PCPI, while failing closed unless its eta-one
   sufficient statistics are numerically equivalent to the shared target.
4. Preserve every P3I.3 dataset, seed, role, budget, utility, quadrature rule,
   threshold, assessment boundary, tie break and held-out restriction.

P3I.4 remains failure-informed development and is not independent
confirmation.
