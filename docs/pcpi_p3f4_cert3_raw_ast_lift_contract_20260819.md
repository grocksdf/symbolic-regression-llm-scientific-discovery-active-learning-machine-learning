# P3F.4-CERT.3 exact semantic-core to raw-AST lift

Status: **EXACT STATIC LIFT IMPLEMENTED; RESIDENT INTEGRATION STILL BLOCKED**

Reviewed GitHub baseline:
`eeb1fbaf1b9e82568caf1ed9c341d3fb114d30d4`

Predecessor development commit:
`7ff83ebdccd2379b9fefadbf9702487e1695604a`

This phase closes the representation mismatch identified by the independent
CERT.2 proof review.  It does not run an experiment, inspect any response,
change the frozen posterior, or compose the new primitive with resident SMC.

## 1. The blocker being solved

The CERT.2 envelope anchor was defined on a hybrid state space:

- exact polynomial-semantic classes for raw AST sizes at most `J`; and
- individual raw ASTs for sizes greater than `J`.

The resident local and reversible-jump kernels act on raw ASTs at every size.
Kernel invariance on two different measurable spaces does not imply invariance
of their composition.  The missing operation was therefore an exact
disintegration of each finite semantic class back onto the original raw-AST
prior, not a new regularizer, empirical correction, or result-dependent rule.

## 2. Original target and quotient map

Let `A` be the countable set of registered raw ASTs, `|T|` the node count, and
`kappa(T)` the exact integer-polynomial key.  The frozen grammar prior is

\[
p(T)=(1-\rho)\rho^{|T|-1}/N_{|T|},
\]

where `N_s` is the exact number of raw ASTs in size shell `s`.  For a cutoff
`J`, define

\[
A_{J,k}=\{T\in A: |T|\le J,\ \kappa(T)=k\},
\qquad
C_s(k)=\#\{T: |T|=s,\ \kappa(T)=k\}.
\]

The exact core class mass is

\[
w_J(k)=\sum_{s=1}^{J}
(1-\rho)\rho^{s-1}\frac{C_s(k)}{N_s}.
\]

This is the push-forward of the original grammar prior.  It is not a uniform
prior over semantic classes.

## 3. The exact conditional lift

For a selected semantic core class `k`, define

\[
r_J(T\mid k)=
\mathbf 1\{T\in A_{J,k}\}\frac{p(T)}{w_J(k)}.
\]

Then

\[
\sum_{T\in A_{J,k}}r_J(T\mid k)
=\frac{1}{w_J(k)}
\sum_{s\le J}C_s(k)\frac{(1-\rho)\rho^{s-1}}{N_s}
=1.
\]

Equivalently, the exact draw is:

1. draw size `s` with probability
   `(1-rho) rho^(s-1) C_s(k) / (N_s w_J(k))`;
2. draw uniformly among the `C_s(k)` raw ASTs of that size and key.

No response, likelihood value, dataset identity, seed selection, fitted
threshold, or posterior result appears in this conditional law.

## 4. Measure-preserving raw-state theorem

Let `d` denote the registered discrepancy component.  Every component design
and collapsed marginal likelihood is a function of `(k,d)`, not of the raw
serialization inside `A_{J,k}`.  Write this class-constant value as `m(k,d)`.

The CERT.2 hybrid core proposal has mass

\[
q_H(k,d)=\frac{w_J(k)p(d)m(k,d)}{C_{J,\lambda}^{\mathrm{RE}}}.
\]

Lift it by retaining `d` and drawing `T` from `r_J(.|k)`.  For every core raw
state,

\[
\begin{aligned}
q_A(T,d)
&=q_H(\kappa(T),d)r_J(T\mid\kappa(T))\\
&=\frac{p(T)p(d)m(\kappa(T),d)}
        {C_{J,\lambda}^{\mathrm{RE}}}\\
&=\frac{\gamma_A(T,d)}{C_{J,\lambda}^{\mathrm{RE}}}.
\end{aligned}
\]

The tail branch already draws a raw AST from the grammar prior conditional on
`|T| > J`; it is unchanged.  Hence the lifted envelope proposal is a proposal
on one common raw-AST/component space and preserves the CERT.2 domination
ratio.  The Mengersen--Tweedie independence-Hastings minorization theorem can
then be applied on that common space, subject to the remaining implementation
and kernel-invariance gates.

## 5. Exact semantic unranking

The implementation does not use rejection from an entire size shell.  Such a
method would be exact only in principle and could have arbitrarily poor
acceptance for a rare semantic key.  Instead it provides a bijection

\[
U_{s,k}:\{0,\ldots,C_s(k)-1\}\longrightarrow
\{T:|T|=s,\kappa(T)=k\}.
\]

At each recursive node, the rank interval is partitioned using the same exact
integer multiplicities as the semantic dynamic program:

- unary `neg` has weight `C_(s-1)(-k)`;
- an ordered `add` derivation `(a,b)` has weight `C_l(a) C_r(b)` when
  `a+b=k`;
- an ordered `mul` derivation `(a,b)` has the same product weight when
  `ab=k`.

After a binary branch is selected, Euclidean division of its local rank by the
right-child multiplicity gives the unique left and right child ranks.  By
induction, every raw AST has exactly one rank and every rank produces exactly
one raw AST with the requested size and key.

Addition derives the right key by exact subtraction.  For a nonzero
multiplication target, each nonzero candidate factor has at most one quotient
because the integer polynomial ring is an integral domain.  The code performs
exact sparse multivariate polynomial division and admits a pair only when the
integer remainder is zero.  The zero polynomial is handled separately so that
the union `left=0 or right=0` counts the double-zero pair once.

## 6. Arbitrary-precision integer ticket plan

The runtime grammar parameter is converted from its shortest round-trip
decimal string, which is also the value used in the grammar's stable JSON
identity.  The registered token is treated as an exact rational; the code
fails closed if it does not round-trip to the runtime value.  Thus each
per-AST prior mass

\[
a_s=(1-\rho)\rho^{s-1}/N_s
\]

is an exact rational.  Let `D` be a common denominator and reduce the positive
integers `h_s = D a_s` by their common divisor.  The class ticket count is

\[
H_k=\sum_{s\le J} C_s(k)h_s.
\]

One uniform arbitrary-precision ticket in `{0,...,H_k-1}` selects a size block
and an AST rank.  Every AST in size `s` owns exactly `h_s` tickets, so

\[
\frac{h_s}{H_k}=\frac{a_s}{w_J(k)}=r_J(T\mid k)
\]

exactly.  The byte-rejection draw is not limited to NumPy's signed or unsigned
64-bit integer range.  The deterministic ticket-to-AST map is separately
testable without random frequency checks.

## 7. Response-free verification obligations

The static tests perform only algebraic and finite combinatorial checks:

| Obligation | Exact check |
|---|---|
| DP multiplicity | Group exhaustive small raw shells and compare every `C_s(k)` |
| Unranking | Compare all ranks with all enumerated ASTs for 1D and 2D grammars |
| Conditional normalization | Sum rational ticket mass to exactly `Fraction(1,1)` |
| Original-prior identity | Check `w_J(k) r_J(T|k) = p(T)` for every size block |
| Component extension | Multiply by an arbitrary rational class-constant factor and check raw identity |
| Large integer support | Check endpoint tickets when `H_k` exceeds 64 bits |
| Failure closure | Reject absent keys, malformed keys, and out-of-range ranks/tickets |

These checks do not generate targets, simulate responses, use real or held-out
data, estimate efficacy, or support a superiority claim.

## 8. Relation to current methods

[Identification-aware MCMC](https://arxiv.org/abs/2511.12847) likewise uses
known observational-equivalence sets to improve movement, but its empirical
performance claims do not establish PCPI's target identity.  PCPI needs the
stronger model-specific equality above: an explicit disintegration that
recovers each raw-AST mass exactly.

[On the lumpability of tree-valued Markov chains](https://arxiv.org/abs/2410.17919)
reinforces that a quotient process requires explicit lumpability conditions.
CERT.3 takes the alternative route authorized by the CERT.2 review: lift the
anchor to the resident raw state space instead of asserting that the existing
local/RJ kernels are lumpable.

The resulting independence proposal remains governed by the domination
condition of [Mengersen and Tweedie
(1996)](https://doi.org/10.1214/aos/1033066201).  Recent finite-sample SMC
results do not remove the need for this target-level proof; they become
relevant only after the raw-state kernel and bridge implementation are frozen.

## 9. Remaining gate before resident integration

This commit intentionally stops before integration.  The following remain
required:

1. construct the complete raw-state envelope proposal over `(T,d)` using this
   lift for the core and the existing exact conditional prior draw for the
   tail;
2. compute proposal and target log masses from one audited parameter identity,
   including component probabilities and the MH correction;
3. prove and test that each composed resident local/RJ kernel has the same raw
   target, bidirectional support, exact forward/reverse probabilities, and any
   required Jacobian;
4. perform an independent source/proof review before enabling composition;
5. only then define a new response-free integration-correctness gate.

No resident SMC, new confirmatory fixture, real dataset, acquisition,
predictive calibration, or held-out path is authorized by CERT.3.
