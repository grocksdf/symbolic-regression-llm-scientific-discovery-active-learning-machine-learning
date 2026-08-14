# PCPI SMC correctness audit

Stage: P2A.1 implementation audit  
Verdict: **P2A.1 exact-reference genealogy/correctness Gate passed**

| Requirement | Canonical implementation | State |
|---|---|---|
| particle `(S, theta, sigma2, log_w, ancestors)` | `pcpi/smc/state.py` | implemented |
| prior initialization | `pcpi/smc/engine.py` | implemented |
| independent collapsed predictive weights | `pcpi/smc/collapsed.py` | implemented/tested |
| adaptive likelihood tempering and CESS | `pcpi/smc/engine.py`, `resampling.py` | implemented/tested |
| ESS and adaptive resampling | `pcpi/smc/resampling.py`, `engine.py` | implemented/tested |
| bridge/parent/child/root genealogy | `pcpi/smc/state.py`, `engine.py` | implemented/tested |
| recomposed genealogy-map validation | `pcpi/smc/state.py` | implemented/tested |
| ESS-only resampling decision audit | `pcpi/smc/engine.py`, `state.py` | implemented/tested |
| arbitrary-bridge invariant rejuvenation | `pcpi/smc/kernel.py` | implemented/tested |
| exact posterior comparison | `pcpi/reference/posterior.py`, `smc/metrics.py` | implemented |
| real measured protocol | `data/real_protocol.py`, `run_pcpi_p2a_real.py` | ready, not run here |
| explicit birth/death/replace support | `pcpi/smc/proposal.py` | implemented/tested |
| forward/reverse proposal probabilities | `pcpi/smc/proposal.py` | implemented/tested |
| MH proposal-ratio correction | `pcpi/smc/kernel.py` | implemented/tested |
| collapsed dimension matching / Jacobian | `pcpi/smc/proposal.py` | explicit unit-Jacobian contract |
| detailed balance / invariant transition | P2B runner and tests | implemented/tested |
| per-move proposal and acceptance telemetry | `state.py`, `engine.py` | implemented/tested |

P2A.1 continues to use its complete default proposal. P2B supplies a frozen,
irreducible proposal graph whose move probabilities can be asymmetric. The MH
acceptance ratio includes `q(reverse)/q(forward)`. Coefficients and variance are
integrated out, so the cross-dimensional state is the collapsed structure
marginal; the auxiliary map is empty and its Jacobian is one. Parameters are
redrawn from the exact final conditional. This validates the finite collapsed
kernel, not arbitrary open-grammar RJ moves.

The formal P2A.1 correctness runner uses an exactly enumerable controlled
fixture, eight seeds, three particle counts, complete failure recording, and a
not-applicable held-out state. The separate real calibration runner requires
official file hashes and keeps held-out closed. Neither result may be used as
scientific-discovery efficacy evidence.

The frozen P2A.1 diagnostic completed 24/24 runs with no failures. Every local
ancestor map recomposed to the recorded root ids; resampling occurred only
below the ESS threshold; conditional ESS remained above 0.79999; the bridge
kernel invariant residual was at most numerical roundoff; and TV/KL errors
decreased with particle count. This closes fixed-universe P2A.1 only.

The P2B Gate uses a separately labelled `inference_correctness_diagnostic_fixture`
with an exactly enumerable seven-structure bank and 24 registered runs. It
checks posterior TV/KL, predictive NLL agreement, weights, CESS, resampling,
genealogy, all move types, proposal normalization, detailed balance, invariant
rejuvenation, particle-count behavior, and seed stability. It cannot be cited
as evidence of scientific-discovery efficacy.
