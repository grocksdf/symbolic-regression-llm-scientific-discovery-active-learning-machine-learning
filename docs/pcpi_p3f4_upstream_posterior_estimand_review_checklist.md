# P3F.4 upstream mathematical-review checklist

Review completed 2026-08-19. The controlling derivations and
counter-certificates are in
pcpi_p3f4_g0_g6_proof_package_20260819.md.

| Gate | Recorded disposition |
|---|---|
| G0 | PROVED after estimand Amendments A1--A2 |
| G1 | PROVED |
| G2 | PROVED; current L=3 bound is decision-useless |
| G3 | PROVED for prior-independence base kernel only |
| G4 | NO-GO for current L=3 certificate |
| G5 | NO-GO for current resident budget |
| G6 | PROVED at the mathematical role-contract level |

Package-wide disposition: **NO-GO; implementation remains blocked.**

This checklist accompanies
`pcpi_p3f4_upstream_posterior_estimand_contract.md`. It is contract-only and
does not authorize inference implementation, fixtures, confirmatory freezing,
predictive calibration, real data, acquisition, or held-out execution.

## Evidence identity

- Source baseline:
  `ffe7239955d9083a7ad6ef878c3213c602027aad`.
- Frozen P3F.3-VR.8 archive SHA-256:
  `16bb9ab7e007fd98614ebb3152f929501db75655882958a1c256770ada6d235a`.
- VR.8 status: final development NO-GO; 24/24 runs completed; five registered
  development decisions failed.
- P3F.3 results must not be re-thresholded or relabelled under P3F.4.

## Review order and kill points

The proof review proceeds in this order. A failure stops the package before
later work.

1. **G0 Estimand scope**
   - Decide whether the scientific claim is explicitly grid-restricted.
   - If it is a continuum claim, supply the response-free
     modulus-of-continuity/coarsening bound.
   - Prove measurability, finiteness, support-extension invariance, and no
     future-response dependence of `C_star`.
   - Define certified numerical bin-boundary handling.
2. **G1 Full target**
   - Prove the countably open grammar/latent prior is proper.
   - Write the unnormalized full-target density and evidence.
   - Recover finite-slice evidence before conditional renormalization.
3. **G2 Posterior tail**
   - Derive a computable upper bound on omitted unnormalized evidence.
   - Show the resulting posterior-tail bound is finite and non-vacuous on the
     entire registered correctness domain.
   - Stop the full-open claim if only a grammar-prior tail is available.
4. **G5 Decision budget**
   - Freeze the scientific loss and acquisition utility.
   - Derive the admissible total-variation or joint-law error from Bayes-risk
     and action-regret budgets.
   - Allocate the budget among estimand coarsening, posterior tail,
     finite-particle bias, island randomness, and numerical evaluation.
5. **G3 Open reversible kernel**
   - Write exact post-filtering forward and reverse probabilities for every
     typed birth, death, and replacement move.
   - Prove support, irreducibility on registered finite slices, detailed
     balance, and stationarity.
6. **G4 Feynman--Kac path**
   - Define the population relative-ESS/chi-square condition.
   - Derive a finite-particle confidence rule rather than relying on plug-in
     CESS alone.
   - Specify fail-closed behavior with no forced terminal beta step.
7. **G6 Data roles**
   - Prove that the target, class map, proposal learning, grids, bin count, and
     thresholds cannot access development outcomes, real validation,
     acquisition responses, or held-out state.

## Required review disposition

For every Gate, record exactly one of:

- `PROVED`: every stated obligation has a checkable derivation;
- `REVISE`: the mathematical object remains coherent but the derivation is
  incomplete; or
- `NO-GO`: the object or required non-vacuous bound does not exist under the
  current target.

Implementation is eligible only when G0--G6 are all `PROVED`. That eligibility
permits only a correctness implementation and response-free fixtures; it does
not permit confirmatory or downstream execution.

## Non-negotiable rejection checks

- Prior tail is not posterior tail.
- Finite-slice agreement is not a full-open posterior certificate.
- Exact polynomial identity is not operational predictive equivalence.
- A sampled-tree clustering is not support-extension invariant.
- ESS, CESS, genealogy, or evidence unbiasedness alone is not a posterior-law
  error bound.
- Island replication does not remove finite-particle bias.
- A learned, variational, LLM, genetic, or controlled proposal does not define
  the posterior and is unusable without exact correction.
- A grid-restricted estimand is not described as continuum equivalence without
  a coarsening proof.
- Numerical intervals crossing a class boundary fail closed.
- No new mechanism code is written before all proof dispositions are recorded.
