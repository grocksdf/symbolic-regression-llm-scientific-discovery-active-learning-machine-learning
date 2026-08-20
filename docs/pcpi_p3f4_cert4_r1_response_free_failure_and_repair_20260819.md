# P3F.4-CERT.4-R1 response-free check failure and test-contract repair

Status: **FIRST WINDOWS CHECK ATTEMPT FAILED; TEST-CONTRACT REPAIR ONLY**

Failed source commit: `8b18f74fbbe4152093db4952d0ab72c4a148308f`

Returned archive:

- file: `p3f4_cert4_response_free_checks_FAILED_20260819.zip`
- SHA-256: `A37597003BB55A9642760A6E901D0C4FB62612E7665ECAC6E8386E4C8D4D9BB9`
- retained attempt status: `failed`

The returned failure is permanent evidence for that source/test identity. Its
`attempt.json` and traceback must not be overwritten, deleted, or relabelled
as a passing run.

## 1. Exact failure

`test_complete_anchor_normalizes_and_recovers_every_core_raw_target_mass`
contained three platform-fragile assertions:

```text
selection_probability_sum == 1.0
selection_normalization_error == 0.0
maximum_log_mass_identity_error < 8e-15
```

On the user's Windows canonical interpreter, the observed values were:

```text
selection_probability_sum             0.9999999999999998
selection_normalization_error          2.220446049250313e-16
maximum_log_mass_identity_error        1.2434497875801753e-14
```

On the development Linux interpreter at the same source identity, they were:

```text
selection_probability_sum             1.0
selection_normalization_error          0.0
maximum_log_mass_identity_error        5.329070518200751e-15
```

The differing results arise after `exp`, `logaddexp`, `log`, and finite
summation of the same positive categorical weights. They are within ordinary
binary64 rounding and do not indicate lost support, target drift, or a failed
MH correction.

## 2. Why the test was wrong

The production builder already accepted an explicit `identity_tolerance`,
defaulting to `2e-12`, and failed closed when either

```text
selection_normalization_error > identity_tolerance
maximum_log_mass_identity_error > identity_tolerance
```

The test nevertheless imposed bit-exact normalization and an unrelated
`8e-15` ceiling. Thus the test asserted a stronger, unstated cross-platform
property than the implementation contract it was meant to verify.

The failure does not authorize choosing a new tolerance from the Windows
values. In particular, CERT.4-R1 does not adopt the post-result value
`2e-14`. It names and reuses the pre-existing `2e-12` contract so the builder,
mass evaluator, and tests share one frozen identity.

## 3. Authorized repair

CERT.4-R1 makes only the following source-level corrections:

1. expose `P3F4_RAW_STATE_ANCHOR_IDENTITY_TOLERANCE = 2e-12`;
2. use it as the default in anchor construction and raw-state mass evaluation;
3. replace bit-exact floating assertions with zero-relative-tolerance checks
   against that same constant; and
4. preserve exact equality for rational component probabilities, conditional
   priors, AST ranks, semantic keys, and support identities.

No target mass, proposal probability, envelope, component prior, cutoff,
response, seed, threshold, dataset rule, regularizer, or resident kernel is
changed.

## 4. Recheck identity and claim boundary

The R1 check is a new response-free source identity and must write to a new
non-overwriting evidence directory. It is not a rerun of an efficacy,
confirmatory, real-data, or held-out experiment. The original failed attempt
remains intact.

An R1 pass establishes only that the static anchor checks express the already
implemented finite-precision contract portably. It does not change the
resident composition NO-GO: the resident `int64` support ceiling, noncanonical
raw evaluation, and absent raw-AST local/RJ proposal remain unresolved.
