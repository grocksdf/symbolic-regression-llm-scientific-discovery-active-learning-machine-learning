# P1 decision — exact finite-bank posterior

Status: **implementation retained; prior generated-data evidence revoked**

The analytic Normal–Inverse-Gamma finite-bank engine, sequential update,
posterior predictive, numerical integration cross-check, and operational-class
pushforward remain production code. Previous generated-observation efficacy
packages are revoked and cannot support a real-data or discovery claim.

P1 is validated through two explicitly separated channels:

1. a fully specified exact-reference fixture recorded only as
   `inference_correctness_diagnostic_fixture` evidence, plus function-level
   regression tests; and
2. exact posterior calculations on provenance-verified measured observations
   inside the P2A real run.

The first channel may support normalization, sequential/batch identity,
parameter integration, and numerical correctness. It is not efficacy evidence.

No claim of scientific efficacy, unknown-law discovery, SMC correctness, EIG,
motif safety, held-out confirmation, or VED follows from P1 alone.
