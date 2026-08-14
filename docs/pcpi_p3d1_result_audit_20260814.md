# P3D.1 certified reference-dominance result audit — 2026-08-14

Decision: **CONTROLLED CORRECTNESS GATE PASSED; REAL INTEGRATION AND EFFICACY
REMAIN UNAUTHORIZED.**

## Frozen evidence identity

- Stage: `P3D.1`
- Experiment: `certified_reference_dominance_correctness`
- Fixture role: `inference_correctness_diagnostic_fixture`
- Source identity: clean Git worktree
- Source commit: `5d71f588398daac3a7c8d982ec3eac0b5834d73c`
- Git tree: `eb8e62cff7f97e30e87123441aa0edaae6aaa3a6`
- Formal source-tree hash:
  `7fbc9cef48bc3b8db4378e36d062bade7faf4d1415e369169d84f8717fc05c4d`
- Production-code hash:
  `77ae08157421018fa896692f304d9f8fe843d7497a30bc5e7d5ce78b5224cd3d`
- Canonical config hash:
  `a8533a79c29071bbca07913040609b70ecb1625e07a3c11616005c6aea894390`
- Runtime dependency-environment hash:
  `03f682959e86595ba29a709b5587efff56910203ef8daa4c5a6bbcad8efc829b`
- Output:
  `outputs/p3d_1_reference_dominance_correctness_5d71f58_20260814`

## Gate result

All 14 frozen decisions passed:

1. exact discrete class-EIG equals direct expected entropy reduction;
2. EIG is nonnegative;
3. EIG respects the class-entropy capacity bound;
4. reference utility is the probability-weighted action utility;
5. probability-weighted reference bounds contain the exact reference utility;
6. separated intervals authorize the targeted handover;
7. every targeted fixture decision exactly dominates the reference utility;
8. overlapping intervals execute the registered reference policy;
9. zero class capacity executes the reference policy without a fitted
   threshold;
10. a fixed registered seed gives a deterministic reference draw;
11. the alternate registered seed changes only the reference draw;
12. stable candidate identity is invariant to candidate-array permutation;
13. class-label and outcome-label permutations preserve EIG;
14. malformed inputs fail closed.

The EvidenceRegistry verifies with one event. The evidence-export manifest
hash matches the run manifest, and all seven listed file hashes were
independently recomputed from the returned output.

## Supporting validation

- P3D.1 plus the isolated inference/integrity regression subset: `69 passed`.
- Python syntax preflight: `114` files, zero failures.
- production static audit: `56` Python files and `14,017` lines, zero failures.
- Full test collection remains blocked with 14 import errors because the
  public Git import is missing the six manifest-listed
  `hypothesis_mvp.data` files. This is a source-completeness failure, not a
  P3D.1 test failure.

## Statistical claim boundary

The returned evidence supports only this conditional implementation statement:

> Given valid simultaneous bounds for frozen-class EIG, the implemented rule
> hands over to a target-seeking action only when its lower bound strictly
> exceeds the registered reference policy's upper bound; otherwise it samples
> from that reference policy.

It does not validate a production interval constructor. It does not repair
posterior misspecification, establish realized no-harm, integrate the rule into
the real acquisition runtime, or support CCPP/Gas efficacy, held-out, motif,
VED, open-grammar, intervention, or scientific-law claims. The run records
`heldout_opened=false`, `selection_used_heldout=false`, and
`formal_efficacy_evidence=false`.
