# P3F.4-CERT.6 resident common-target adapter contract

Status: **RESIDENT SUPPORT/TARGET ROOT REPAIRS PASS; REJUVENATION IMPORT BLOCKED**

GitHub baseline: `e4cf80b76fc834859e68bc6e594740223b136f53`

This phase repairs the two resident source boundaries diagnosed by CERT.4 and
composes their response-free representation with the standalone CERT.5
involutive transition. It does not import that transition into resident
`_rejuvenate`, call `ScalableOpenTargetSMC.run`, materialize a response, or
execute a simulated, formal, real, validation, acquisition, confirmatory, or
held-out experiment.

## 1. Scope and serial Gate

CERT.5 proved a Markov kernel on the complete raw product state

\[
z=(T,d)\in\mathcal A\times\mathcal D
\]

with complete bidirectional support, exact pathwise proposal mass and a
semantic-only target evaluator. It deliberately left the resident engine
unchanged. CERT.6 has exactly three source obligations:

1. replace every resident fixed-shell/open-prior raw-AST rank draw that could
   exceed a machine integer by exact arbitrary-precision byte rejection;
2. make the resident collapsed design and discrepancy basis a function only of
   `(polynomial_key, component_state_id)`; and
3. expose one fail-closed adapter that evaluates the proved CERT.5 proposal on
   this canonical resident target.

Actual rejuvenation import is a later Gate. This prevents a partially repaired
resident path from being executable merely because its standalone algebra
passes.

## 2. Arbitrary-precision resident support

For shell size `s`, let `N_s` be the exact Python integer returned by
`grammar.expression_count(s)`. The new primitive samples

\[
R\sim\operatorname{Uniform}\{0,\ldots,N_s-1\}
\]

by masked byte rejection and calls the already proved raw-AST unranking
bijection. No value is converted to NumPy `int64`. The complete open prior
first samples the geometric size using the registered rational continuation
probability and then performs the same exact shell draw.

Resident `particle._sample_expression_of_size` and the `maximum_nodes=None`
branch of `sample_open_prior_expression` now delegate to these primitives.
The response-free proof reaches size 29 of the two-feature grammar, where

\[
N_{29}=75{,}182{,}313{,}393{,}693{,}794{,}847>2^{64}.
\]

The archived NumPy failure witness remains in the CERT.4 test. The same test
now also proves that the repaired resident path returns a valid size-29 AST;
the negative evidence is preserved rather than overwritten.

Discrete discrepancy-component draws in the resident prior path use the exact
integer-ticket component plan from CERT.4. Thus the resident initial-state
support and the CERT.5 auxiliary component support have the same registered
identity.

## 3. Canonical resident target

For every raw AST, compute the exact polynomial key

\[
k=\kappa(T).
\]

`build_resident_semantic_design` accepts only `(k,d)` together with the frozen
actions and target contract. It constructs

\[
X_{k,d}=
\begin{cases}
f_k, & d=\text{none},\\
[f_k\;A_{k,d}], & d\ne\text{none},
\end{cases}
\]

where `f_k` is evaluated from the exact key and `A_{k,d}` is the registered
projected discrepancy factor. Both base-design and basis caches are keyed by
the semantic class identifier, never by `raw_ast_id`.

Consequently raw aliases `T_1,T_2` with
`kappa(T_1)=kappa(T_2)` use bit-identical design, discrepancy basis, prior
precision and collapsed marginal-likelihood computation for a fixed
component. Their target masses may still differ through the legitimate raw-AST
prior factors `p_G(T_1)` and `p_G(T_2)`.

This repairs the CERT.4 counterexample `1` versus `(x+1)+(-x)` at `x=1e16`
at the target-definition boundary. It does not introduce a tolerance or round
the raw evaluations into agreement.

## 4. Common-target transition adapter

`resident_common_target.py` binds three frozen identities:

- the resident target contract hash;
- the grammar hash; and
- the CERT.5 local/RJ plan hash.

Given a CERT.5 proposal path, it evaluates both endpoints through the callable

```text
semantic_log_marginal_evaluator(polynomial_key, component_state_id)
```

and no raw serialization is present in that callable interface. The target is

\[
\gamma(T,d)=p_G(T)p_D(d)m(\kappa(T),d),
\]

and the adapter returns

\[
\log\alpha=
\min\{0,\log\gamma(z')-\log\gamma(z)
          +\log q(u'\mid z')-\log q(u\mid z)\}.
\]

The reverse proposal is reconstructed and checked against the same plan.
Therefore the augmented forward and reverse accepted log flows agree to the
pre-registered CERT.4/CERT.5 `2e-12` floating identity tolerance; all support,
rank, prior and endpoint identities remain exact integers or `Fraction`.

The construction follows Green's reversible-jump auxiliary-state identity and
the involutive formulation used in iMCMC:

- Green (1995):
  https://people.maths.bris.ac.uk/~mapjg/papers/RJMCMCBka.pdf
- Neklyudov et al. (2020):
  https://proceedings.mlr.press/v119/neklyudov20a.html
- Saad et al. (2023), subtree-replace involution:
  https://proceedings.mlr.press/v202/saad23a.html
- official AutoGP source inspected for the task-independent involutive pattern:
  https://github.com/probsys/AutoGP.jl

These sources motivate the transition construction; they do not transfer an
empirical result, grammar, likelihood, truncation or tuning rule into PCPI.

## 5. Response-free checks

The CERT.6 runner executes the retained 22 CERT.3--CERT.5 proof checks plus
seven new checks:

| Obligation | Check |
|---|---|
| Arbitrary precision | fixed-shell, exact open-prior and resident size-29 draws avoid `int64` |
| Semantic identity | raw aliases construct identical base designs and discrepancy bases |
| Resident source path | `_make_particle` caches by semantic class rather than raw AST |
| Adapter balance | exact endpoints, proposal ratio, reverse map and accepted flow agree |
| Gate identity | mismatched plan hashes and attempted SMC authorization fail closed |
| Target API | raw evaluators and non-finite semantic masses are rejected |
| Execution isolation | the adapter has no particle import, SMC construction or `.run()` call |

The runner reports `29/29` only after compiling every repository Python file
outside `.git`, `.venv` and `evidence`.

## 6. Authorization boundary

CERT.6 authorizes the following source-level objects only:

- arbitrary-precision resident raw-AST prior/shell sampling;
- canonical semantic resident design/basis construction; and
- the standalone resident common-target transition adapter.

It hard-codes:

```text
resident_rejuvenation_import_authorized = false
resident_smc_integration_authorized = false
resident_smc_invoked = false
```

The resident `_rejuvenate` method still selects only its historical global
independence/uniform mechanisms and does not call the adapter. The next
admissible phase is a separate response-free source-composition proof that
imports this exact adapter into `_rejuvenate`, verifies the actual resident
endpoint reconstruction and acceptance path, and keeps SMC execution blocked
during proof. No calibration, real data, acquisition, held-out, confirmatory,
efficacy or paper-superiority claim is authorized by CERT.6.
