# P3F.4-CERT.22 operational identity and scale preflight

Status: frozen response-free **NO-GO** candidate; identity-bound user Gate
pending.

## Finding

CERT.21 correctly left operational execution closed. The required next audit
cannot authorize it: the certified object and the registered real task are not
the same object.

The frozen P3F.4 target configuration has one feature. The complete executable
CERT.20/21 source-composition fixture has one feature and cutoff `J=1`, with
six core/component target balls. Its production configuration mentions
`J=17`, but no executable Gate constructs the complete `J=17` source table.

The registered real protocol instead contains:

| Target | Features | Seeds | Initial H0 size |
|---|---:|---:|---:|
| CCPP/PE | 4 | 8 | 32 |
| Gas Turbine/CO | 9 | 8 | 32 |
| Gas Turbine/NOX | 9 | 8 | 32 |

Thus operational execution requires 24 separately identified H0 artifacts.
No such artifact hash is currently bound, and CERT.13 intentionally accepts
only algebraic fixtures. CERT.22 does not read real data to manufacture those
identities.

## Response-free scale lower bound

For cutoff `J`, every monomial through degree `floor((J+1)/2)` occurs in the
grammar: a degree-k monomial needs `k` variable leaves and `k-1`
multiplications. At `J=17`, stars-and-bars gives at least
`C(d+9,9)` distinct semantic classes. With the three registered discrepancy
components, this implies:

| Dimension | H0 count | Classes/H0 (minimum) | Balls over family (minimum) |
|---:|---:|---:|---:|
| 4 | 8 | 715 | 17,160 |
| 9 | 16 | 48,620 | 2,333,760 |
| Total | 24 | — | 2,350,920 |

This is a strict lower bound, not a cost estimate. General integer-coefficient
polynomials add semantic classes beyond the monomial subset. The exact raw-AST
recurrence gives 5,480,405,422,085 expressions through `J=17` for `d=4` and
1,331,131,316,840,170 for `d=9`; semantic aggregation avoids raw enumeration
but does not prove the complete class table feasible.

No wall-clock or byte figure is inferred from the tiny fixture. Without a
full-scale constructor, measuring that fixture would not certify operational
time or storage.

## Gate semantics

The expected successful Gate status is `passed-no-go`. It proves that the
identity mismatch and scale blockers are detected before H0, system entropy,
output creation, acquisition, validation or heldout access. It does not prove
efficacy and does not authorize a real run.

The preflight derives a deterministic output identity from the H0-family and
frozen target/source/runner configuration identities. This identity is a
reservation only. No file or directory may be materialized while the decision
is NO-GO.

## Root repair

The next implementation must bind the target dimension to every registered
task and eliminate eager complete-core materialization from the operational
path. A lazy or streamed exact construction must preserve the declared target,
semantic lift, tail support, rejection domination and no-retry semantics. It
must then receive response-free, full-dimension time and storage certification.
Sample counts, thresholds, datasets, seeds and `J=17` may not be reduced to
make the audit pass.

Real H0 values remain closed until that source-level repair and a subsequent
identity Gate pass.
