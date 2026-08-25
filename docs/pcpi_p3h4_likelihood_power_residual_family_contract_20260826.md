# PCPI P3H.4 likelihood-power residual-family contract

## Decision

P3H.4 freezes a separate prequential residual state for every preregistered
likelihood-power posterior target. A residual law calibrated or reconstructed
under one target may not be copied, indexed into, or replayed under another.

The registered method is
`model-specific-likelihood-power-prequential-raw-pit-family-v1`. This is a
state-mapping correctness phase. It does not run a real acquisition or claim
that the likelihood-power family is calibrated.

## Why a single shared residual state is invalid

Let \(\mathcal E=\{\eta_1,\ldots,\eta_K\}\subset(0,1]\) be the likelihood-power
ambiguity set fixed before acquisition. For the same opened history
\(\mathcal H_{t-1}\), different powers generally give different generalized
posteriors and predictive CDFs:

\[
F_{\eta,t-1}(y\mid x,\mathcal H_{t-1}).
\]

Consequently, the raw residual coordinate for opened observation \(t\),

\[
U_{\eta,t}=F_{\eta,t-1}(Y_t\mid X_t,\mathcal H_{t-1}),
\]

is target-specific. Except at special coincidences such as the common prior
forecast at the first observation, \(U_{\eta,t}\ne U_{\eta',t}\). Copying a
nominal residual state to all powers therefore applies a correction learned in
the wrong probability coordinate. It also breaks the one-to-one pairing
required by the P3H.3 transformed utility.

P3H.2 does not remove this issue. Its compatibility result concerns the
registered \(\eta=1\) structurewise discrepancy forecast. The robust
acquisition family uses finite-bank `SequentialReferencePosterior(eta)`
targets. P3H.4 explicitly records
`p3h2_eta1_calibration_transferred_to_family: false`.

## Frozen chronological construction

The ambiguity powers are unique and stored in strictly increasing order. They
must share one structure bank and one frozen design transform. The power set is
fixed independently of acquisition responses.

The opened history order is
`initial-development-stored-order-then-acquisition-reveal-order`. Initial
development observations retain their stored post-subset order. Each later
observation is appended only after the pool oracle has revealed the selected
response. Dataset row order, response value, current result and numerical
convergence may not reorder this history.

For every \(\eta\), reconstruction starts from both the \(\eta\)-posterior
prior and an empty KT dyadic residual state. At position \(t\), it executes:

1. recover \(P_{\eta,t-1}\) from exactly the strict prefix
   \(\mathcal H_{t-1}\);
2. evaluate the unclipped raw PIT \(U_{\eta,t}\) using that target's predictive
   CDF;
3. append \(U_{\eta,t}\) only to target \(\eta\)'s residual state;
4. admit \((X_t,Y_t)\) to target \(\eta\)'s posterior history.

The implementation currently reconstructs strict prefixes by exact conjugate
batch fitting. This is intentionally simple and auditable. It is algebraically
equivalent to sequential conjugate updates but costs \(O(Kn^2)\) prefix work.
No efficacy or timing claim is attached to this correctness implementation;
an optimized incremental implementation must prove state identity before it
can replace the frozen route.

## Binding and response erasure

For each target, the family bundle contains:

- likelihood power and posterior target hash;
- the exact engine object;
- the exact posterior object fitted to the complete opened history;
- the target-specific raw-PIT state and predictive residual law;
- a shared chronological-history commitment hash.

The production scorer accepts the family bundle, not a tuple of unlabelled
residual laws. It requires the acquisition `PosteriorModel` engine and
posterior objects to be identical to those bound in the corresponding family
state. A separately refitted, numerically equivalent posterior is rejected.
This prevents ordering mistakes, stale states and cross-power substitution.

After reconstruction, raw action and response arrays are erased from the
bundle. The scorer receives only posteriors, residual states/laws, observation
count and commitments. The commitment includes the ordered opened history but
does not expose it. This is response isolation, not a claim that hashes make
sensitive data anonymous outside the registered experiment boundary.

## Correctness decisions

The deterministic P3H.4 Gate checks:

1. strict increasing power order;
2. distinct posterior target hashes and target-specific PIT sequences;
3. bitwise repeatable reconstruction;
4. history-order sensitivity of residual states;
5. order equivalence of the final conjugate batch posterior;
6. isolation of an already-issued prefix family from unpassed future values;
7. erasure of raw history arrays from the acquisition bundle;
8. acceptance only of exact engine/posterior object bindings;
9. absence of RNG, candidate-response, result-status and threshold branching.

The hand-authored algebra fixture is not simulated data and does not estimate
performance. Its power-separation threshold is a floating-point correctness
guard, not an empirical tuning constant.

## Leakage boundary

Only already-opened development and acquired responses may enter family
reconstruction. At acquisition round \(r\), the current candidate pool,
validation responses, held-out responses and all future oracle responses are
absent. Candidate covariates may enter the later P3H.3 utility only after the
family state has been frozen for that round.

No retry may rebuild the same round after observing a failed candidate score,
candidate response or held-out result. A reconstruction or binding failure is
terminal for that run and must be published as negative evidence.

## What a pass does not establish

A P3H.4 pass establishes only the state map and its software isolation. It does
not establish:

- that any \(F_{\eta,t}\) is calibrated;
- that the KT residual innovation model is adequate for every \(\eta\);
- that P3H.2 compatibility transfers from its structurewise \(\eta=1\) target;
- that family-wide P3H.3 rankings are numerically resolved on real pools;
- acquisition efficacy, held-out improvement, discovery or paper acceptance.

Before real acquisition, a separate frozen calibration-only Gate must assess
the complete model-specific family on real opened calibration sequences, with
candidate responses and held-out data closed. Only after that result is audited
may a unique operational command be considered.
