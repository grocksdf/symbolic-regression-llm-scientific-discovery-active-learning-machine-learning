# P3M.2 action-conditional lifecycle composition (2026-09-07)

P3M.2 composes the P3M.1 conditional residual law into the complete frozen
likelihood-power family.  Each power owns its own strict-prefix action/PIT
state because each generalized-Bayes forecast induces different raw PITs.
All four states use the same response-free context transform and bandwidth
rule, but never share response-derived counts.

At a query, every candidate action is passed explicitly to its local residual
law.  Each ambiguity model returns mean information, lower-tail entropy-gain
CVaR and negative-gain mass from one normalized candidate-specific joint.  The
selection score is the lower envelope of CVaR across the four powers; mean
information is audit-only.  The existing covariate-only representative guard
and interval ranking rule remain unchanged.

One typed decision freezes the selected candidate identifier, action and
prior state hash.  Only one matching reveal is admitted.  Before appending the
new action/PIT pair, the update evaluates the selected action's prefix law,
applies exactly those class factors to the posterior, advances the conjugate
state, and verifies its class marginal against direct Bayes updating.  Then
all four conditional residual states advance exactly once.

The formal query identity now has a separate P3M schema and hashes the exact
candidate action matrix.  Changing one floating bit changes the identity.
This is essential because the candidate actions now affect the residual law,
not merely the base predictive components.

P3M.2 remains a no-data correctness/source-composition stage.  Durable
candidate-specific chunk checkpoints are still blocked.  Therefore no formal
real run is authorized by this stage.
