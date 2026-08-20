# P3F.4-CERT.4 resident kernel audit

Status: **FINITE INDEPENDENCE FORMULAS PASS; RAW LOCAL/RJ COMPOSITION NO-GO**

Reviewed baseline: `6410f1e509d90c656a3a32cc113c16deeb776265`

Audited source:

- `hypothesis_mvp/pcpi/open_target/particle.py`
- `hypothesis_mvp/pcpi/open_target/rjmcmc.py`
- `hypothesis_mvp/pcpi/smc/proposal.py`
- `hypothesis_mvp/pcpi/smc/kernel.py`

The audit is response-free. It does not call `ScalableOpenTargetSMC.run`,
materialize a likelihood fixture, or execute any formal, simulated, real,
validation, confirmatory, acquisition, or held-out experiment.

## 1. Decision table

| Kernel surface | Bidirectional support | Proposal ratio | Target invariance | CERT.4 decision |
|---|---:|---:|---:|---|
| `open_target/rjmcmc.py` finite exhaustive reference | PASS | PASS | PASS | Reference-only; not scalable raw local/RJ |
| Resident finite-slice prior independence | PASS | PASS | PASS | Admissible on its declared finite component support |
| Resident finite-slice complete uniform | PASS | PASS (`0`) | PASS | Admissible on its declared finite component support |
| Resident independent prior/uniform mixture | PASS | PASS | PASS | Admissible on its declared finite component support |
| Resident random scan of prior and uniform MH kernels | PASS | Per selected kernel | PASS | Convex combination of invariant kernels |
| Countably-open resident prior sampler | **FAIL** | Not globally defined | Not established | `int64` support ceiling |
| Resident raw-state target vs semantic core target | **FAIL** | Different target identities | Not composable | Raw evaluation is not class-constant |
| Resident raw-AST local subtree/RJ kernel | **ABSENT** | Absent | Absent | Move labels are diagnostics, not a local/RJ proposal |
| P2B finite-bank birth/death/replace catalog | PASS on finite bank | PASS | PASS on finite bank | Different state/model; cannot be reused as raw-AST proof |

The overall resident-composition Gate is **NO-GO**. Passing finite-support
independence formulas does not cure either countable-support failure or target
identity mismatch.

## 2. Finite proposal algebra that passes

For resident prior independence,

\[
q(y\mid x)=p(y),
\qquad
\log\frac{q(x\mid y)}{q(y\mid x)}=\log p(x)-\log p(y).
\]

Because the target log mass is `log p(z) + log m(z)`, the prior terms cancel
and the implemented MH ratio is correct.

For complete uniform on a finite registered component set of size `K`,

\[
q(y\mid x)=1/K,
\]

including self proposals, so the proposal log ratio is zero.

For the independent mixture,

\[
q(y)=\eta p(y)+(1-\eta)/K,
\]

and resident code uses the full mixture probability in both directions. For
the random-scan kernel mixture, it first chooses one individually reversible
kernel and applies that kernel's own MH ratio. A convex combination of these
invariant kernels remains invariant. The static algebra test builds all four
transition matrices from arbitrary positive mass vectors and checks
row-stochasticity, detailed balance, and stationarity.

These results cover the formulas actually present in `_rejuvenate`; they do
not establish an unimplemented typed-subtree RJ move.

## 3. Blocker RKA-1: countable support is truncated by an integer API

`_sample_expression_of_size` draws a shell index with
`numpy.random.Generator.integers(total)`. For the registered two-feature
grammar,

\[
N_{29}=75{,}182{,}313{,}393{,}693{,}794{,}847,
\]

which has 67 bits. Calling the resident sampler at size 29 raises

`ValueError: high is out of bounds for int64`.

Every finite size has positive probability under the geometric open prior.
Therefore the implemented resident proposal is not defined on the declared
countably-open support, and a global reverse-support claim is false. The same
issue existed in the old CERT.2 tail helper; CERT.4 fixes it inside the
standalone anchor through arbitrary-precision unranking, but deliberately does
not modify resident SMC in this phase.

## 4. Blocker RKA-2: resident likelihood is not exactly class-constant

The semantic core requires

\[
m(T,d)=m(\kappa(T),d)
\]

for every raw serialization in one exact polynomial class. Resident
`_make_particle` instead evaluates each raw operator tree through
`evaluate_expression` and caches by `raw_ast_id`.

The response-free counterexample is

\[
T_1=1,
\qquad
T_2=(x+1)+(-x).
\]

Both have the same exact polynomial key. At `x=1e16`, raw IEEE-754 evaluation
gives `T1=1` and `T2=0`, whereas canonical polynomial-key evaluation gives
`1`. The resulting base designs, projected discrepancy spaces, and collapsed
likelihoods can therefore differ inside one semantic class.

This is not repaired by a numerical tolerance: the anchor and resident code
currently define different finite-precision targets. Composing their kernels
would invalidate the CERT.3 lift identity and the anchor minorization theorem.

## 5. Blocker RKA-3: no resident raw local/RJ proposal exists

The P3F.3 contract permits future typed subtree birth/death/replacement moves,
but resident `_rejuvenate` currently chooses only:

- full-state prior independence;
- full-state complete uniform on a finite slice;
- their independent mixture; or
- their random-scan kernel mixture.

Diagnostic labels such as `within-equivalence-class` and
`cross-equivalence-class` classify the result of a global proposal. They do
not define a local tree edit, a forward/reverse neighborhood probability, an
auxiliary-variable map, or a Jacobian.

The P2B `StructureProposalCatalog` does implement audited birth/death/replace
edges, but only between a frozen finite `ReferenceBank`. Its state, prior, and
proposal graph are not the P3F raw typed-AST target, so its proof cannot be
transferred by name.

## 6. Required root repair before integration

The next admissible phase is source-level correctness work, not an experiment:

1. define one canonical raw-target design from the exact polynomial key and
   use the same class identity for resident design/basis caching;
2. replace every countably-open resident raw-AST draw with the CERT.4
   arbitrary-precision size/rank construction while retaining exact proposal
   mass;
3. decide honestly whether the scalable kernel remains global independence or
   adds a typed-subtree local/RJ proposal;
4. if local/RJ is added, derive every move-type probability, site-selection
   probability, subtree law, reverse move, support condition, and collapsed
   unit-Jacobian statement before implementation;
5. pass a new response-free common-target transition audit before the anchor
   is imported by resident SMC.

No threshold, regularizer, seed, response, evaluation result, or dataset rule
enters these repairs.

## 7. Theory boundary

The exact semantic disintegration remains compatible with the principle of
[identification-aware MCMC](https://arxiv.org/abs/2511.12847), but that paper's
performance results do not prove PCPI's raw-target identity. The distinction
between exact and approximate projection is consistent with current work on
[tree-valued chain lumpability](https://arxiv.org/abs/2410.17919). Once a
common target and full proposal support exist, the anchor's global domination
can use the independence-Hastings result of [Mengersen and Tweedie
(1996)](https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/Rates-of-convergence-of-the-Hastings-and-Metropolis-algorithms/10.1214/aos/1033066201.short).

Until then, no resident target-invariance, mixing, SMC fidelity, calibration,
or downstream scientific claim may cite CERT.4 as an integrated result.
