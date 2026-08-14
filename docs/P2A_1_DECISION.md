# P2A.1 decision — robust target-preserving fixed-universe SMC

Status: **EXACT-REFERENCE SMC GENEALOGY/CORRECTNESS GATE PASSED**

## Why this substage exists

The frozen P2A real result passed exact finite-bank posterior agreement, but
Gas Turbine CO/NOX ended with only 1–9 original root ancestors at 2,048
particles. That is a warning about irreversible lineage loss, not evidence
against the concentrated finite-bank posterior. P2A.1 addresses the warning
before any trans-dimensional proposal is introduced.

## Frozen statistical construction

For every finite-bank structure, coefficients and noise variance are
analytically integrated. The independent SMC tracker computes

\[
g_t(S)=p(y_t\mid x_t,S,H_{t-1}).
\]

If one observation is too informative for a stable update, the target is
bridged through

\[
\eta_{t,\beta}(S)\propto p_{t-1}(S)g_t(S)^\beta,
\qquad 0=\beta_0<\cdots<\beta_B=1.
\]

The next temperature is selected by conditional ESS with a frozen target of
0.8. Every bridge uses normalized incremental weights. Systematic resampling
is now triggered only when the post-update ESS crosses the frozen threshold;
a nonterminal temperature alone never forces resampling. A symmetric MH
structure kernel targets every current bridge. The transition matrix is checked against its stationary
distribution. At the final bridge, coefficients and noise variance are drawn
from their exact conditional Normal–Inverse-Gamma posterior, reconstructing
joint posterior particles without using plug-in likelihood weights.

The implementation is independent of dataset id, filename, target name, and
the observed validation result. Its only task-dependent input is the visible
numeric design matrix and the dimension-derived finite reference bank.

Every bridge records local ancestor indices, parent particle ids, fresh child
particle ids after resampling, and the complete root-ancestor vector. The Gate
recomposes those maps from the initial population and checks that recorded root
ancestry is exact and nonincreasing. Rejuvenation never resets a root id.

## Frozen P2A.1 correctness Gate

The primary Gate is a 24-run, exactly enumerable controlled diagnostic with
three particle counts and eight frozen seeds. It must satisfy:

- batch and sequential posterior agreement;
- normalized weights and valid ESS/CESS;
- unbiased-resampling smoke behavior;
- resampling if and only if ESS crosses the configured threshold;
- complete parent/child/root genealogy-map consistency;
- invariant bridge rejuvenation;
- systematic particle-count convergence in TV/KL;
- predictive NLL and log-evidence agreement;
- all seeds retained, with no replacement.

This fixture is labelled `inference correctness diagnostic fixture` and cannot
support a real-data or scientific-discovery efficacy claim.

The frozen run completed 24/24 registered runs with zero failures and no seed
replacement. At 2,048 particles, mean structure TV was 0.02001, mean
structure KL was 0.001900, mean predictive-NLL error was 0.002977, and mean
absolute log-evidence error was 0.03444. All genealogy maps recomposed exactly,
every resampling decision matched the ESS threshold, and the final root
ancestor fraction was 0.3662–0.3906 across seeds. All Gate decisions passed.

## Heldout-closed real calibration

The registered CCPP and grouped Gas Turbine CO/NOX calibration should complete all
72 preregistered runs with no replacement seeds. It must retain the original
P2A exact-agreement thresholds and additionally satisfy:

- every observation reaches temperature 1;
- minimum conditional-ESS fraction is at least 0.79999;
- every resampling event retains at least 25% distinct immediate parents;
- no immediate parent produces more than 25% of the population;
- every realized bridge kernel has invariant residual at most \(10^{-12}\);
- held-out remains closed and selection never uses held-out.

The real calibration remains useful external evidence, but it does not replace
the exactly enumerable correctness Gate. Its genealogy thresholds are generic safeguards against instantaneous
collapse. They were frozen before the P2A.1 real result and are not selected
from dataset outcomes. Long-horizon root ancestry remains a reported
diagnostic because ordinary particle genealogies can coalesce even when the
current target approximation is correct.

## Decision boundary

The passed Gate supports:

> P2A.1 implements a Rao–Blackwellized, adaptively tempered, target-preserving
> finite-universe SMC with complete bridge and genealogy telemetry agrees with
> its exact reference posterior under the controlled correctness diagnostic.

It does not support open-grammar or trans-dimensional correctness, real-data
efficacy superiority, motif safety, held-out confirmation, or VED claims. P2B
is unblocked for revalidation against this new canonical source identity.
