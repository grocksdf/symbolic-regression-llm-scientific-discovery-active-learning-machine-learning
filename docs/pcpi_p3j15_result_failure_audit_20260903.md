# P3J.15 completed real-development result audit

## Immutable execution identity

The completed P3J.15 output is
`outputs/p3j15_formal_real_acquisition_1a2f5eac_20260902`. Its manifest binds
source commit `1a2f5eac0ef0bd0955e9fa900effe588d1a3f70c`, source tree
`d3d0a965b664a8db015913cceda26b4d15282c89`, configuration SHA-256
`15111d61cbf653ca1a0f69af9deaf8b9afb3f49f4e8b97ade309c9b24ec856c3`,
and Python executable SHA-256
`560b9ef7d856608ab8da02ded2dc8a1951ad1f424c382c0ec6a698874165a18e`.

The result hashes are:

- `summary.json`:
  `0c4b67bfddcd0b12152cc35e2ad833f56de549862fd45aba2e9c636a781cd702`;
- `RUN_MANIFEST.json`:
  `ab9e716ec7004c1c814ff4252145f9e72bc31f1cc0f5ae37740e5540006476bb`;
- `tables/paired_effects.csv`:
  `a025e885acafcfeb0f034fec03fd2e41d1ee9891291446f7e6ef25d006695c6d`.

All 96 registered runs completed with zero failures. The protocol Gate passed,
all P3J decisions reported certified rankings, and held-out remained closed.
This is valid formal protocol evidence.

## Effectiveness assessment

The registered assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED` and
`formal_efficacy_evidence` is false. Against random, mean PCPI-minus-baseline
frozen-class entropy gain was `-0.1001485` on CCPP with 95% interval
`[-0.4165440, 0.2162470]`, and `-0.0358589` for the pooled Gas family with
interval `[-0.2828288, 0.2111111]`. Neither family establishes a positive
primary effect. The corresponding class-gain negative-transfer rates were
`0.5` and `0.625`, so the registered negative-transfer control also failed.

Secondary predictive nAULC was not uniformly better. PCPI-minus-random was
`-0.0113754` on CCPP but `0.0213727` for the pooled Gas family, where negative
values favor PCPI. Comparisons against uncertainty and QBC likewise failed the
requirement of nonpositive mean delta in every dataset family. These secondary
metrics cannot rescue the failed primary decision.

The experiment therefore supports only the statement that the registered
transactional, leakage-closed P3J protocol was executable and auditable. It
does not support superior acquisition, stronger scientific discovery, or a
held-out claim.

## Root-cause implication

The negative result is not repaired by relaxing significance criteria,
changing seeds, selecting a favorable dataset, or adding a score penalty.
P3J's unrestricted counterfactual residual density for each scientific class
creates a structural identifiability problem: every sufficiently rich class
can transport its own predictive CDF to the same observed response law. P3J
also mixed normalized acquisition predictive densities with unnormalized
powered-likelihood structure updates for `eta<1`.

P3K.1 addresses those two probability-model defects under a new method and
stage identity. P3J.15 and this audit remain immutable negative development
evidence and must not be overwritten or reclassified after the repair.
