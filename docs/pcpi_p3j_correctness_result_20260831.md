# PCPI P3J.1/P3J.2 correctness result

P3J.1 and P3J.2 passed their response-free correctness and source-composition
Gates on 2026-08-31. No real, candidate-response, validation-response,
held-out or simulated efficacy data were accessed, and operational execution
remains unauthorized.

P3J.1 passed eight frozen decisions. Uniform per-class residual laws recovered
the base Student-t class-EIG within `2e-5`. In the hand-authored algebra
fixture, the base action scores were `[0.6484314, 0.0018987]`, while the
nonidentity class-conditional scores were `[0.5356750, 0.5884826]` with
safety-inflated fine/coarse differences `[0.0159545, 0.0011619]`; the changed
ranking was separated under those diagnostic envelopes. This fixture proves
implementation sensitivity and normalization only, not real-data efficacy.

P3J.2 passed eight frozen lifecycle decisions. It requires exactly the powers
`[0.125, 0.25, 0.5, 1.0]`, erases raw initial response history after state
reconstruction, binds scoring to those exact calibrated states, and admits
only one response matching the frozen state hash, candidate ID and action.
Every target then advances exactly once without retry.

The first lifecycle test exposed and closed an important claim boundary.
For `eta=1`, the calibrated reporting posterior is numerically required to
equal the direct class Bayes update under `p(c)q_c(y)`. For `eta<1`, the base
targets are likelihood-power generalized-Bayes sensitivity models, whose
one-step update is not ordinary Student-t predictive Bayes. They therefore use
the same pre-reveal class calibration factor on top of their own generalized
update but are not called ordinary Bayesian posteriors. Only `eta=1` is the
scientific/reporting posterior.

The focused P3J suite passed 12/12. The source-integrity test initially rejected
the new public scorer because it exceeded the repository's 100-line audit
limit; validation and result construction were split into independently
auditable helpers without changing behavior, after which integrity plus P3J
passed 13/13. The complete repository then passed all 785 collected tests.

The restored runtime identity is CPython 3.11.9 with dependency hash
`b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`,
identical to the frozen canonical environment. P3J still has no production
runner dispatch, no measured-data configuration and no execution command.
