# PCPI source completion assessment

Current candidate stage: P3B.10 representative-safe maximin joint acquisition.

| Capability | State |
|---|---|
| finite-bank conjugate posterior | implemented and exact-reference checked |
| adaptive fixed-universe SMC | P2A.1 Gate passed |
| corrected finite collapsed trans-dimensional kernel | P2B Gate passed |
| operational predictive classes | P3A.2 exact-reference checked |
| Gauss--Jacobi class-EIG | P3A.2 numerical Gate passed |
| SafeBayes generalized posterior | implemented and controlled |
| posterior/predictive coordinate consistency | P3B.6 controlled Gate passed |
| budget-resolved operational classes | P3B.7 controlled Gate passed |
| joint class--predictive acquisition | P3B.8 controlled Gate passed; real efficacy failed |
| representative-safe joint acquisition | P3B.9 controlled and real protocol passed; real efficacy failed |
| representative-safe maximin joint acquisition | P3B.10 implemented; controlled 27-decision Gate pending formal identity-bound run |
| matched-budget real acquisition | P3B.9 protocol-valid but efficacy-insufficient; P3B.10 rerun blocked on correctness |
| EvidenceRegistry | one hash-chained registry per formal run |
| held-out confirmation | closed/not attempted |
| motif target invariance | absent; excluded from core |
| arbitrary open-grammar RJ-SMC | absent |
| VED | not started |

The production tree contains one posterior, one SMC implementation, and one
acquisition implementation. Diagnostic fixtures are role-labelled and cannot
support real efficacy claims.

P3B.9 completed 96/96 returned runs with an intact evidence chain and held-out
closed, but significantly underperformed random in CCPP frozen-class gain even
though the representative guard was active on every query. P3B.10 preserves
that guard and ranks PCPI candidates by the minimum joint utility over the four
likelihood powers frozen before the repair. The nominal posterior remains the
reporting/evaluation posterior. The repair adds no dataset formula, response
direction, result-derived threshold, new budget, or held-out input.

P4/P5 remain blocked until the P3B.10 controlled Gate and a new heldout-closed
real run both pass and are independently audited.
