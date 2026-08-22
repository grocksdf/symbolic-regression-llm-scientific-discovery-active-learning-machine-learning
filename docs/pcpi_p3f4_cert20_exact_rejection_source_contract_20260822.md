# P3F.4-CERT.20 exact-rejection source contract

Status: frozen response-free source candidate; identity-bound user Gate pending.

## 1. Purpose and claim boundary

CERT.19 proved the exact envelope-rejection law and the fixed-candidate
binomial decision algebra, but accepted abstract atom bounds only. CERT.20
binds that theorem to the implemented open-target source:

1. final-`H0` CERT.14 collapsed log-marginal Arb balls;
2. CERT.17's frozen `512 * 2^r` precision schedule and CERT.18's actual
   exact-input evaluator;
3. CERT.3's exact conditional semantic-core raw-AST lift;
4. CERT.4's exact component and analytic-tail prior draws; and
5. one explicitly accepted external ideal independent-byte premise.

This is source composition, not an experiment. Operational `H0` target
access, real data, heldout data, acquisition, resident SMC and confirmatory
materialization remain false.

## 2. Actual core atom masses

For semantic class `k` and component `d`, CERT.14 returns an outward dyadic
interval for the final-`H0` log marginal likelihood,

\[
  \ell_{k,d}\in[\underline\ell_{k,d},\overline\ell_{k,d}].
\]

CERT.20 exponentiates the two exact endpoints with Arb directed rounding and
then multiplies by the exact rational semantic-class and component priors:

\[
[\underline h_{k,d},\overline h_{k,d}]
=p_0(k)p_0(d)
[\exp(\underline\ell_{k,d}),\exp(\overline\ell_{k,d})].
\]

After multiplication the endpoints remain exact rationals but need not be
dyadic. The implementation therefore distinguishes a CERT.14
`CertifiedDyadicInterval` from a CERT.20 `CertifiedRationalInterval`; it does
not round a non-dyadic prior product back onto a binary grid.

The builder requires exactly one ball for every registered semantic-core and
component pair, one common target/provider identity, the last frozen `H0`
index and bridge beta exactly one. Missing, duplicated, crossed or non-final
balls fail before a proposal source is constructed.

## 3. Exact proposal source

The core atom ticket probabilities are the exact CERT.19 dyadic fractions.
A fixed-width byte block selects one of all `2^b` tickets without floating
normalization or modulo reduction. Conditional on a selected core atom, the
existing semantic lift draws exactly from the raw grammar prior restricted to
that semantic class.

The unique tail atom first draws the registered component prior with exact
integer tickets and then draws from the infinite raw-AST prior conditional on
`|T|>J`. The tail target/proposal ratio cancels both raw-AST and component
prior factors, leaving

\[
  \rho^J L(T,d)/q_{\mathrm{tail}}.
\]

For a core atom the same cancellation leaves

\[
  p_0(k)p_0(d)L(k,d)/q_{k,d}.
\]

Division by CERT.19's exact domination constant `B` gives an outward rational
acceptance-probability interval. Any refined upper endpoint above one is a
domination violation and fails closed.

## 4. Exact lazy uniform comparison

CERT.15--18 used one fixed 256-bit cell plus an allocated unresolved
probability. That finite-cell path is suitable for a bounded-error SMC
comparison ledger, but conditioning an exact rejection sample on numerical
non-abort could distort its accepted law. CERT.20 therefore registers a
separate exact-lazy comparison rather than silently calling an unresolved cell
an exact Bernoulli draw.

At round `r`:

1. evaluate the actual CERT.14/18 target at precision `512 * 2^r`;
2. intersect it with the preceding valid rational enclosure;
3. only then read the next fixed 256 ideal bits; and
4. decide only if the complete uniform prefix cell lies below or above the
   complete probability enclosure.

The precision values and bit-block size are frozen; no midpoint, tolerance,
endpoint surrogate or result-derived precision value is introduced. The
number of revealed blocks is lazy. Under the explicit FLINT inclusion and
pointwise convergence premise already recorded by CERT.18, and the newly
accepted ideal continuous-uniform premise, the intervals converge to `p` and
`P(U=p)=0`; hence the comparison terminates almost surely and equals
`1{U<p}` exactly.

This new exact-lazy source does not change CERT.17's older statement that
adaptive threshold extension was unauthorized for the CERT.16 finite-grid
ledger. It reuses the registered numerical schedule in a separately named
algorithm whose need follows from exact rejection sampling.

## 5. Accepted external randomness premise

CERT.20 explicitly accepts the model

`iid-uniform-bytes-independent-across-all-coordinates`.

Its materialization calls `python secrets.token_bytes`, backed by the
operating-system cryptographic random source. This is an external modelling
premise, not a theorem proved by inspecting Python, the operating system or
hardware. The source records both of the following as false:

- physical independence proved by source inspection; and
- a deterministic PRNG promoted to a mathematical ideal product law.

The premise is now visible and auditable instead of being smuggled in through
distinct seeds or counter addresses.

## 6. Selection, confirmation and proposal caps

The production source freeze is
`configs/p3f_4_cert20_exact_rejection_source_freeze.json`. Before any new
response it fixes:

- semantic core cutoff `J=17`;
- 32 proposal ticket bits;
- proposal-cap failure budget `1/100` and the CERT.19 exact-integer Chernoff
  formula;
- an independent 8192-accepted-draw selection pilot;
- empirical-mode candidate selection with registered-class-ID lexicographic
  tie breaking;
- confirmation stages `512, 2048, 8192, 32768`;
- MAP regret budget `1/50` and familywise confirmation error `1/20`; and
- disjoint logical coordinate domains for selection and confirmation.

The numerical proposal cap is derived after the frozen `H0` target balls give
the certified acceptance lower bound. This is a theorem-mandated response-
frozen budget calculation, not result-based performance tuning. A cap miss
erases the incomplete accepted-state buffer and returns abstention. Terminal
states cannot be extended, retried, reseeded or partially read.

The candidate selector consumes exactly the frozen pilot count and returns one
candidate. It cannot fall back to a second candidate. Confirmation uses only
fresh draws from its disjoint coordinate domain.

## 7. Deterministic checks

The CERT.20 tests establish:

1. explicit acceptance and honest limitation of the ideal-byte premise;
2. complete binding of actual CERT.14 balls to the core/component grid;
3. outward exponentiation and exact non-dyadic rational prior products;
4. exact dyadic ticket selection and both exact conditional lifts;
5. actual CERT.18 512/1024-bit rejection-boundary refinement;
6. evaluator-before-bits ordering and exact extreme-prefix decisions;
7. erasure, abstention and retry prohibition at the proposal cap;
8. no result access before a frozen fixed-candidate boundary;
9. deterministic independent-pilot mode selection and tie breaking; and
10. equality between the production freeze and implemented defaults.

These are algebraic and source checks. They use no simulated observations,
real data, heldout data, acquisition outcomes or efficacy measurements.

## 8. Remaining boundary after a passing Gate

A passing identity-bound CERT.20 Gate closes the rejection-source
composition, not the experiment. The next source phase must build the guarded
operational runner that:

1. materializes separate selection and confirmation transcripts from the
   accepted source premise;
2. evaluates the complete `J=17` final-`H0` core table once;
3. executes proposal and exact-lazy comparison loops under the two frozen
   caps;
4. emits a complete evidence ledger or one indivisible abstention; and
5. remains unable to access acquisition or heldout outcomes.

Only after that runner has its own passing response-free Gate may a real
experiment command be issued to the user.
