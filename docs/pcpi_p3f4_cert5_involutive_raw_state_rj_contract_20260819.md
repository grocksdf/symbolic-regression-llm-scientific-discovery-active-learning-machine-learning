# P3F.4-CERT.5 exact involutive raw-state local/RJ contract

Status: **STANDALONE RAW-STATE LOCAL/RJ KERNEL PROVED; RESIDENT INTEGRATION BLOCKED**

GitHub baseline: `63d3d47c6bf6bf2402ca23cd545487a1b57abaf0`

This phase constructs the missing raw typed-AST local/RJ proposal on the same
countably infinite state space as CERT.4. It is response-free and does not
import, modify, or invoke `ScalableOpenTargetSMC`. It does not materialize a
response, simulated experiment, formal experiment, real dataset, validation
role, confirmatory fixture, acquisition state, or held-out information.

## 1. Primary-source construction

The construction follows the reversible auxiliary-map requirement of Green's
RJMCMC and the discrete involution formulation of iMCMC:

- Green (1995), *Reversible jump Markov chain Monte Carlo computation and
  Bayesian model determination*:
  https://people.maths.bris.ac.uk/~mapjg/papers/RJMCMCBka.pdf
- Neklyudov et al. (2020), *Involutive MCMC: a Unifying Framework*:
  https://proceedings.mlr.press/v119/neklyudov20a.html
- Saad et al. (2023), *Sequential Monte Carlo Learning for Time Series
  Structure Discovery*, especially its self-reversing SUBTREE-REPLACE move:
  https://proceedings.mlr.press/v202/saad23a.html
- official AutoGP implementation, inspected at
  `probsys/AutoGP.jl@2ad372db2b8bea176e3b32a769bfb5d6cff80866`:
  https://github.com/probsys/AutoGP.jl

PCPI does not copy AutoGP's grammar, depth controls, parameter proposals,
likelihood, or empirical settings. It adopts only the task-independent
involutive subtree-regeneration pattern and supplies PCPI-specific exact raw
prior and component masses.

## 2. State and exact auxiliary law

The state remains

\[
z=(T,d)\in\mathcal A\times\mathcal D,
\]

where `T` is one registered raw typed AST and `d` is the spike or one
registered discrepancy-kernel component. For a tree with node-address set
`A(T)`, the auxiliary variable is

\[
u=(a,S,d'),
\qquad
q(u\mid z)=\frac{1}{|T|}p_{\mathcal G}(S)p_D(d').
\]

The implementation draws all three discrete choices exactly:

1. `a` is an arbitrary-precision uniform ticket over the deterministic
   preorder addresses of `T`;
2. the new subtree size is drawn from the exact rational geometric law, and
   its shell rank is drawn by arbitrary-precision byte rejection; and
3. `d'` is drawn from the CERT.4 exact integer-ticket component prior.

No NumPy bounded-integer API is used. Therefore every finite subtree and every
registered component has positive auxiliary probability, including shells
whose raw-AST count exceeds 64 bits.

The law is normalized analytically:

\[
\sum_{a\in A(T)}\frac1{|T|}
\sum_{S\in\mathcal A}p_{\mathcal G}(S)
\sum_{d'\in\mathcal D}p_D(d')=1.
\]

## 3. Involution and complete support

Let `T[a <- S]` replace the subtree at address `a`. If `T_a` is the discarded
subtree, define

\[
F\big((T,d),(a,S,d')\big)
=\big((T[a\leftarrow S],d'),(a,T_a,d)\big).
\]

The same address exists after replacement, all ancestors and siblings are
unchanged, and applying `F` twice recovers the original state and auxiliary
variable. The state and auxiliary spaces are discrete; all continuous
coefficients and discrepancy coordinates remain collapsed. Consequently the
continuous auxiliary dimension is zero and

\[
\log |J_F|=0.
\]

Complete one-step support follows from the root address. For any raw states
`(T,d)` and `(T',d')`, choose `a=()` and regenerate exactly `(T',d')`.
The forward event has positive mass, and the reverse root event regenerates
`(T,d)` with positive mass. Non-root addresses retain genuinely local grow,
prune, replace, component-refresh, and self moves.

## 4. Exact proposal ratio

For one realized auxiliary path, write

\[
T'=T[a\leftarrow S].
\]

The implementation retains, as exact `Fraction` values,

\[
q_f=\frac1{|T|}p_{\mathcal G}(S)p_D(d'),
\qquad
q_r=\frac1{|T'|}p_{\mathcal G}(T_a)p_D(d).
\]

The acceptance calculation uses

\[
\log(q_r/q_f)
=\log q_r-\log q_f
\]

without assuming cancellation. The address is part of the auxiliary identity:
the implementation does not incorrectly replace this pathwise probability by
an unproved aggregate over multiple addresses that may reach the same tree.

## 5. Common target and invariance

The target mass is

\[
\gamma(T,d)=p_{\mathcal G}(T)p_D(d)m(\kappa(T),d).
\]

The target evaluator receives only the exact polynomial key `kappa(T)` and
component identifier. Raw AST serialization is absent from its interface, so
class-constant evaluation is structural rather than a floating tolerance.

For the realized involutive edge, the acceptance probability is

\[
\alpha(z,u)=\min\left\{1,
\frac{\gamma(z')q(u'\mid z')}{\gamma(z)q(u\mid z)}
\right\}.
\]

The forward augmented flow is therefore

\[
\min\{\gamma(z)q(u\mid z),\gamma(z')q(u'\mid z')\},
\]

which is identical in the reverse direction. Summing paired auxiliary events
proves detailed balance and target invariance for the state kernel on the
countable raw space.

## 6. Response-free proof checks

The CERT.5 runner executes 22 registered checks: the five CERT.3 lift checks,
seven CERT.4 anchor checks, three retained resident NO-GO checks, and seven new
local/RJ checks.

| Obligation | Check |
|---|---|
| Address map | Every node has one preorder path; replace/discard is an involution |
| Complete support | Every pair of small raw states/components has a positive root edge and exact reverse |
| Open support | A deterministic ticket source reaches a 67-bit shell without `int64` |
| Common target | Algebraic aliases expose the same semantic evaluator input |
| Proposal ratio | Grow, prune, replace, component and self edges satisfy pathwise detailed balance |
| Invariance | A finite censored reference matrix is stochastic, reversible and stationary |
| Failure closure | Invalid paths/components, non-finite target masses and endpoint mismatch fail closed |

All discrete probability identities, support identities, inverse mappings and
auxiliary normalization checks use exact integers or `Fraction`. Floating
log-flow and matrix residuals reuse the already registered CERT.4 `2e-12`
identity tolerance; CERT.5 does not select a new tolerance from its outputs.

During development, the first finite reference mistakenly allowed regenerated
subtrees only through size two while its state shell included size-three
discarded subtrees. The matrix correctly failed reverse support. The proof
fixture was repaired at the cause by making its auxiliary shell cover every
subtree that can be discarded from the finite state shell. No tolerance,
target, threshold, response, seed, or result direction was changed.

## 7. Authorization boundary

CERT.5 authorizes only `raw_state_local_rj.py` as a standalone exact kernel
construction. `resident_smc_integration_authorized` remains hard-coded false.

The resident engine still calls its old NumPy `int64` prior sampler and raw
IEEE-754 expression evaluator, and it does not yet call this kernel. Therefore
resident SMC composition, simulated/formal/real experiments, confirmatory
materialization, acquisition, held-out access, and paper efficacy or
superiority claims remain blocked. The next admissible step is a separate
source-level common-target integration audit, not an experiment.
