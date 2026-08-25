# P3H.3 transformed-law acquisition correctness result

## Outcome

The frozen P3H.3 correctness Gate passed on source commit
`ef8ffcf94f55ecfc1f37877acc0e4916da2be1bc` and source tree
`668630d0a006b990dc86fc32f12634511ff15d1b`.

All nine registered decisions passed:

- the finite-grid outcome marginal equals the P3H residual-law quadrature
  measure;
- the projected class marginal equals the frozen operational-class posterior;
- mutual information is nonnegative and bounded by frozen class entropy;
- the identity residual law contains the independently integrated Student-t
  EIG reference inside the registered numerical envelope;
- a nonidentity residual law changes the utility;
- positive affine transformations of the response leave the utility unchanged;
- a separated ranking resolves;
- an overlapping ranking returns no selection;
- production source contains no response, RNG, real-data or old-EIG dispatch
  surface.

The terminal status is `passed`. It is an acquisition-utility correctness
diagnostic, not a real or simulated experiment.

## Numerical audit

The two action scores under the nonidentity residual fixture were
`[0.561504759786561, 0.001795277918044989]`. The independently integrated base
Student-t scores were `[0.6484314363752692, 0.0018986645931942095]`; the maximum
change was `0.08692667658870823`, confirming that the old Student-t EIG was not
silently reused.

The maximum response-grid row-marginal error was
`3.469446951953614e-17`. The maximum frozen class-marginal error was
`1.7108536809473662e-13`, below the preregistered `2e-13` projection threshold.
The log-domain projection used at most 193 iterations. Positive affine response
invariance held to `2.220446049250313e-16`.

Under the identity residual law, absolute errors against the independent
infinite-domain Student-t integration were
`[1.1784669981418361e-05, 1.1685440874213433e-05]`, inside the preregistered
fine/coarse-plus-reference envelopes
`[0.0025896303110112396, 0.00014583373058120995]`. These are numerical
containment decisions, not claims that the envelopes are rigorous finite-order
bounds.

The separated fixture had interval gap `0.5588763895188514` and resolved its
first action. The identical-action fixture had gap
`-5.413637078444111e-05` and returned no selection. The production route is
also tested to terminate on overlapping P3H intervals rather than fall back to
the legacy posterior-variance or Student-t utility.

## Integrity and regression

The result used CPython 3.11.9 and runtime dependency hash
`b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`, exactly
the frozen P3G.5/P3H.2 dependency identity. The production source SHA-256 was
`f891d1c2091c036c11547598c84191256e021ceda924875c9c52697a7f0fe779`.

The archived output directory is
`outputs/p3h_3_semiparametric_acquisition_correctness_ef8ffcf9_20260825`.
Its `summary.json` SHA-256 is
`6b1c03bb1b54bfb862caa51b8883191fa2b7ba150a871079ec744f06886d9f32`, exactly
matching `manifest.json`.

Before the frozen Gate, the P3H.3 targeted suite passed 13/13. The prior
acquisition suite passed 63/63 and the discrepancy suite passed 8/8. The full
repository regression collected and passed 674/674 tests with exit code zero.
No failure was masked or relaxed.

## Access and claim boundary

The Gate records:

- `simulated_experiment: false`;
- `real_data_access: false`;
- `validation_response_access: false`;
- `candidate_response_access: false`;
- `candidate_selection_executed: false`;
- `operational_execution_authorized: false`;
- `heldout_access: false`;
- `formal_efficacy_evidence: false`.

A pass establishes the registered finite-grid algebra, leakage isolation,
identity/nonidentity behavior and fail-closed numerical ranking behavior. It
does not establish residual stationarity, posterior adequacy, strict
finite-order ranking correctness, acquisition efficacy, held-out improvement,
scientific discovery or an AISTATS result.

No real acquisition command is authorized by P3H.3. The next admissible design
work must freeze how each likelihood-power ambiguity model obtains its own
prequential residual state; one nominal residual law may not be silently shared
across models.
