# P3F.4-CERT.23 lazy complete-prior rejection source

Status: frozen response-free source candidate; identity-bound user Gate
pending.

## Root repair

CERT.22 proved that an eager `J=17` semantic-core/component table does not bind
the registered `d=4/9` real tasks and has a minimum of 2,350,920 target balls.
CERT.23 removes that table rather than making it faster or reducing `J`.

Let the raw state be `s=(T,d)`, with exact registered prior `p(s)`, collapsed
likelihood `L(s)`, evidence `Z=sum_s p(s)L(s)`, and a certified global envelope
`M >= L(s)` for every state. CERT.23 proposes

`s ~ p(s)`

and accepts with probability

`a(s)=L(s)/M`.

The joint probability of proposing and accepting `s` is `p(s)L(s)/M`.
Conditioning on acceptance therefore gives exactly `p(s)L(s)/Z`, the declared
posterior on the complete countably-open raw state space. No semantic quotient
or finite cutoff enters this identity.

## Exact proposal

The raw AST draw uses the registered geometric law on every positive node
count and arbitrary-precision exact unranking within the selected shell. The
component uses the exact spike/kernel integer-ticket prior. Both have complete
support. Tests exercise this same implementation at `d=4` and `d=9`.

Each proposal computes its polynomial key only after the raw state is drawn and
requests exactly one CERT.18 refined target ball. The accept/reject comparison
retains CERT.20's outward exponential and CERT.17 precision schedule. A target
ball above the global envelope fails closed.

## Cap lower bound without a core table

The actual acceptance probability is `Z/M`. A finite set of distinct raw
states gives a rigorous lower bound on `Z`. CERT.23 freezes the terminal ASTs
`one,x0,...,x(d-1)` crossed with all three registered components. This needs
`3(d+1)` target balls: 15 for CCPP and 30 for either Gas target, instead of an
unknown complete polynomial class table.

The sum of their exact prior masses times outward likelihood lower bounds,
divided by the envelope upper bound, is a valid acceptance-probability lower
bound. The existing exact-integer Chernoff formula then freezes selection and
confirmation proposal caps. Cap exhaustion erases partial samples and causes
terminal abstention; it never authorizes retries, smaller samples, a different
anchor family or a changed target.

The algebraic fixture deliberately retains its loose log-envelope value `10`.
It produces selection and confirmation caps of 210,405,467 and 326,128,474.
These are negative feasibility diagnostics, not operational estimates, and
show why source correctness alone cannot authorize execution. CERT.23 does not
tighten that envelope after observing the diagnostic.

## Complexity and claim boundary

Proposal construction has zero eager semantic classes and zero eager core
target balls. Persistent source state contains the immutable plan plus one
raw AST and component per proposal. Target evaluation count is one per
proposal. The anchor preflight is linear in feature dimension.

This is a source-correctness and scalability result, not an operational
feasibility or efficacy result. Operational H0, system entropy, output,
real-data, acquisition, validation and heldout access remain closed. The next
Gate must bind the 24 H0 identities and audit the resulting frozen caps and
measured deterministic kernel cost before any real sampling is authorized.
