# Real-data-only evidence reset

Effective 2026-08-05, formula-generated, simulated, semi-simulated, and
benchmark-generated outcomes are excluded from the paper evidence chain.

This exclusion applies to the empirical conclusions previously associated with
G2.1, G4, G6.x, and G7.x whenever their observations or target instances were
generated. The corresponding code and outputs may remain as audit archives,
but they cannot support tables, figures, abstract claims, RQ answers, or the
conclusion.

The reset does not require a rewrite of the core implementation. The following
components remain reusable engineering infrastructure:

- dataset-role isolation and hidden-label pool acquisition;
- fixed candidate representation and fitting;
- posterior weighting and disagreement calculations;
- matched acquired-label budgets;
- evidence manifests and role audits.

The following empirical components are not reusable as evidence:

- scenario or formula outcome generators;
- `true_class`, true-class mass, and recovery-to-true-class metrics;
- true-structure oracle acquisition;
- artificial measurement-noise injection;
- phase-transition, formula-plus-noise, Feynman, Nguyen, and SR benchmark rows.

The active gate is G4R.1, defined in
`docs/g4r1_real_matched_budget_contract.json`. It uses official, hash-verified
UCI observations and disjoint development/confirmation source rows. NIST ASD
and VED are reserved for later, separately frozen real-data experiments; they
are not used to rescue or tune G4R.1.

Until G4R.1 and the subsequent real-only gates are complete, the current paper
empirical sections must be treated as a legacy draft and not as a
submission-ready evidence statement.
