# P3F.4-CERT.4 complete raw-state envelope anchor

Status: **STATIC RAW-STATE ANCHOR IMPLEMENTED; RESIDENT COMPOSITION NO-GO**

GitHub baseline: `6410f1e509d90c656a3a32cc113c16deeb776265`

This phase completes the CERT.2 envelope proposal on one common raw state
space after the CERT.3 semantic lift. It neither invokes nor modifies resident
SMC, and it does not materialize a response, simulated experiment, real data,
validation role, confirmatory fixture, acquisition, or held-out state.

## 1. Common target identity

The state is

\[
z=(T,d)\in\mathcal A\times\mathcal D,
\]

where `T` is one registered raw typed AST and

\[
\mathcal D=\{\text{none}\}\cup\{\text{registered kernel states}\}.
\]

For one frozen likelihood-power vector, the unnormalized collapsed target is

\[
\gamma(T,d)=p_{\mathcal G}(T)p_D(d)m(T,d).
\]

The component prior is not folded into a semantic-class weight:

\[
p_D(\text{none})=1-\omega,
\qquad
p_D(j)=\omega p_K(j).
\]

`build_raw_state_component_prior_plan` reads the shortest round-trip decimal
tokens already used by the target registry and converts them to exact rational
probabilities. Kernel probabilities must sum to one exactly under that
identity. Arbitrary-precision integer tickets then sample `d` without a
64-bit ceiling.

## 2. Exact core construction

Let `J` be the frozen cutoff and `kappa(T)` the exact polynomial key. The exact
core class mass remains

\[
w_J(k)=\sum_{s\le J} C_s(k)
\frac{(1-\rho)\rho^{s-1}}{N_s}.
\]

The anchor builder accepts an explicit frozen table

\[
\ell_{k,d}=\log m(k,d)
\]

containing exactly one entry for every semantic class at `s <= J` and every
component. Missing entries, extra entries, non-finite masses, or a core value
above the certified likelihood envelope fail closed. The hybrid core atom has
log mass

\[
\log G(k,d)=\log w_J(k)+\log p_D(d)+\ell_{k,d}.
\]

After selecting `(k,d)`, CERT.3 supplies the exact disintegration

\[
r_J(T\mid k)=\frac{p_{\mathcal G}(T)}{w_J(k)},
\qquad |T|\le J,\ \kappa(T)=k.
\]

Therefore the ideal lifted core proposal mass is

\[
q_A(T,d)=\frac{\gamma(T,d)}{C_J}.
\]

## 3. Exact infinite-tail law

Let `rho` be the registered rational continuation probability. Conditional on
`|T| > J`, write `|T|=J+R`, where

\[
P(R=r)=(1-\rho)\rho^{r-1},\qquad r\ge1.
\]

CERT.4 samples this law through exact rational Bernoulli continuation tickets.
It does not call NumPy's geometric or bounded-integer samplers.

Conditional on size `s`, `unrank_raw_expression` implements a bijection

\[
U_s:\{0,\ldots,N_s-1\}\longrightarrow\{T:|T|=s\}
\]

by the exact unary/binary grammar recurrence. The rank is drawn with
arbitrary-precision byte rejection. This removes the `int64` ceiling in the
previous conditional-tail helper and gives every finite raw AST positive
proposal mass.

The exact tail conditional mass returned with a draw is

\[
p_{\mathcal G}(T\mid |T|>J)
=\frac{p_{\mathcal G}(T)}{\rho^J}.
\]

## 4. Normalized envelope proposal

Let `M` be the certified response-energy likelihood envelope and

\[
Z_J=\sum_{k,d}w_J(k)p_D(d)m(k,d),
\qquad
U_J=\rho^J M,
\qquad
C_J=Z_J+U_J.
\]

The complete raw proposal is

\[
q_E(T,d)=
\begin{cases}
p_{\mathcal G}(T)p_D(d)m(\kappa(T),d)/C_J,
& |T|\le J,\\
p_{\mathcal G}(T)p_D(d)M/C_J,
& |T|>J.
\end{cases}
\]

The implemented top-level categorical consists of every `(k,d)` core atom and
one tail branch. Its actual finite-precision selection probabilities are
stored in the plan; the tail owns the exact complement of the cumulative core
intervals. The plan records both normalization error and maximum deviation
from the log-domain ideal identity and fails closed outside the registered
`2e-12` static identity tolerance. This is an honest finite-precision
implementation record, not a claim that arbitrary transcendental likelihood
weights are rational numbers.

## 5. Exact target and proposal log masses used by MH

For every proposed or current raw state, `evaluate_raw_state_anchor_mass`
records

\[
\log\gamma(T,d)
=\log p_{\mathcal G}(T)+\log p_D(d)+\log m(T,d).
\]

For a core state, the implemented proposal log mass is the stored atom
selection log probability plus the exact CERT.3 conditional-lift log mass. A
recomputed core likelihood that differs from the frozen class/component atom
fails closed.

For a tail state, the implemented proposal log mass is the stored tail-branch
log probability plus the exact component and conditional-tail log masses. A
tail likelihood above `log M` fails closed.

The independence-MH correction is exactly

\[
\log\alpha(x,y)=\min\left\{0,
\log\gamma(y)+\log q_E(x)-\log\gamma(x)-\log q_E(y)
\right\}.
\]

No cancellation is assumed by the implementation: both target and actual
proposal log masses are retained. This preserves target correctness even when
top-level categorical probabilities contain audited floating-point rounding.

## 6. Response-free verification

The CERT.4 checks contain no response vector or fitted result. They establish:

| Obligation | Static check |
|---|---|
| Component prior | Exact rational spike/kernel normalization and tickets |
| Raw unranking | Complete small-shell bijection and endpoint ranks in a 67-bit shell |
| Infinite tail | Exact rational geometric law and a shell above the NumPy `int64` ceiling |
| Core identity | Every small raw AST/component satisfies `log q = log gamma - log C` within the recorded arithmetic tolerance |
| Proposal normalization | Enumerated core raw mass plus analytic tail branch equals one |
| MH correction | Pairwise detailed balance for core/core, core/tail, and tail/tail states |
| Sampler/mass coupling | A returned draw carries the same proposal and target masses consumed by MH |
| Failure closure | Incomplete core grids, envelope violations, and non-class-constant core masses are rejected |

These are algebraic/combinatorial correctness checks only. They do not support
posterior fidelity, calibration, efficacy, or scientific-discovery claims.

## 7. Relation to current theory

The domination condition remains the independence-Hastings condition of
[Mengersen and Tweedie (1996)](https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/Rates-of-convergence-of-the-Hastings-and-Metropolis-algorithms/10.1214/aos/1033066201.short).
Recent [identification-aware MCMC](https://arxiv.org/abs/2511.12847) supports
using known observational-equivalence sets for better movement, but it does
not replace PCPI's model-specific measure-preserving lift or its proposal-mass
audit. Work on [lumpability of tree-valued Markov
chains](https://arxiv.org/abs/2410.17919) likewise reinforces that projection
onto a tree quotient needs explicit conditions; CERT.4 stays on the original
raw space instead of assuming resident lumpability.

## 8. Authorization boundary

CERT.4 authorizes only this standalone raw-state anchor and its response-free
checks. `resident_smc_integration_authorized` is hard-coded `False` in the
plan. Composition, SMC execution, simulated/real/held-out experiments,
confirmatory materialization, acquisition, and paper superiority claims remain
blocked by the separate resident-kernel audit.
