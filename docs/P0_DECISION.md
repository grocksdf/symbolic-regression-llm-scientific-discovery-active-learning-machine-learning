# P0 decision — statistical freeze and evidence reset

Status: **contract frozen; generated-data experiment evidence invalidated**

The posterior target remains the joint distribution over symbolic structure,
coefficients, and homoscedastic Gaussian noise under a normalized structure
prior, Gaussian coefficient prior conditional on variance, and Inverse-Gamma
variance prior. The primary estimand is the finite-action operational
predictive-equivalence class; the default terminal loss is 0–1 class loss.

Formal efficacy evidence must use registered, provenance-verified measured
data. A small, fully specified controlled fixture may enter a separate
`inference_correctness_diagnostic_fixture` evidence role for exact posterior,
SMC, or EIG correctness only. Disposable unit-test arrays never enter an
experiment output, Gate, claim-matrix evidence cell, or manuscript result.

Untouched-heldout remains closed. P0 supports the formulation claim only; P2A
requires the frozen local real-data run before any correctness GO decision.
