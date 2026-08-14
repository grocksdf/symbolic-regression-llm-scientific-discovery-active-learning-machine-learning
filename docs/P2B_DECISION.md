# P2B decision — corrected collapsed trans-dimensional SMC

Status: **frozen exact-reference Gate passed; 24/24 runs audited**

## Statistical object

P2B targets the finite-bank collapsed structure posterior after analytically
integrating coefficients and noise. The explicit proposal catalog contains
birth, death, and equal-dimensional replace moves. Every directed edge has
computable forward and reverse probability, declared coefficient dimensions,
reverse support, and a unit collapsed Jacobian. The MH kernel uses the full
proposal-ratio correction.

This is a rigorous finite collapsed trans-dimensional construction. It does
not claim correctness for arbitrary subtree edits, open grammars, LLM token
proposals, or uncollapsed RJ parameter maps.

## Frozen Gate

The formal config runs 8 unchanged seeds at 128, 512, and 2048 particles.
Acceptance is fail-closed and requires all 24 runs, exact-reference agreement,
particle convergence, normalized proposal/kernel rows, reverse support,
detailed balance, target invariance, normalized SMC weights, CESS control,
genealogy diversity, and observed proposals and acceptances for all move types.

The runner prints every run live and writes crash-durable JSONL progress. The
result includes complete failures, bridge diagnostics, ancestor indices,
per-move telemetry, a hash-chained EvidenceRegistry, source/config identities,
tables, and PDF/SVG/PNG figures.

## Evidence classification

The controlled fixture is **DIAGNOSTIC** evidence for RQ2 only. It cannot
support a claim that PCPI discovers stronger laws, outperforms baselines on
real measurements, or discovers a new mechanism. Those claims remain blocked
until the P3 acquisition Gate and matched-budget P5 real experiments.

## Next decision

The frozen P2B Gate passed without seed replacement. P3 operational predictive
classes and exact class-EIG validation are therefore authorized. The result
remains diagnostic RQ2 evidence and must never be presented as real discovery
efficacy.
