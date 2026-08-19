# P3F.4-CERT.2 post-confirmatory numeric ledger

Status: **DEVELOPMENT POSTMORTEM ONLY — NOT CONFIRMATORY EVIDENCE**

Input archive:
`p3f4_certification_confirmatory_20260819.zip`

SHA-256:
`3a6c357946143929c7a2eb8c4dd3404e2435bc0f8ad1f2dca8a05fd6391d45b6`

Frozen source commit:
`5b05e9158b1a550e967a88a3123f76985346fc8e`

The formal CERT.CF.1 decision remains NO-GO. The calculations below apply a
new theorem after responses were opened and therefore cannot replace, amend,
or rerun that decision.

## 1. Independent audit of the frozen result

- ZIP traversal and internal checksum checks passed.
- Config hash:
  `dc89217920cca81fb91a8d25fa2d1bea1e94086e47f886fbe30a9dbf26d6cfca`.
- Certification module hash:
  `73b268c5a998be65c4a8ebc245471c2b1c3d1b54faef0fa06faa86ac749d3e8d`.
- Development runner hash:
  `5dbeee18df3e7d0452034d1febb707d0f8810f93b91f7fa86b25936ea424e6a2`.
- Confirmatory runner hash at the frozen Git commit:
  `ee148fea27fea19445bff42030b6a4bcb7823c48e733fd7e2b987ad07e3606ae`.
- Eight of eight target vectors were reproduced bitwise from the frozen PCG64
  generators, and their hashes were unique.
- All five decisions were independently recomputed from the stored numbers.

Formal decision counts:

| Decision | Passed |
|---|---:|
| Semantic prior mass | 8/8 |
| Likelihood envelope | 8/8 |
| Certified bridge path | 8/8 |
| Posterior-tail ceiling | 6/8 |
| Derived proposal-mixing bound | 6/8 |
| Complete run | 6/8 |

Both failures occurred for the same fixture under both frozen seeds. The old
tail upper was `0.03538557223524339` and `0.029124923973810293`, above the
registered `0.01` ceiling. Because old one-step TV upper equalled tail upper,
the proposal decision was a deterministic descendant of the same failure.

## 2. Frozen constants for the RE calculation

\[
a_0=3,
\quad b_0=0.08,
\quad \rho=0.4,
\quad J=17,
\quad \nu=8.
\]

The old component envelope was

\[
M_8^{\mathrm{flat}}=5639.272478769479,
\]

and the old tail evidence upper was

\[
\rho^{17}M_8^{\mathrm{flat}}
=0.0009688196347819116.
\]

For each stored response vector, the new calculation uses only
\(R=\sum_i y_i^2\), the closed-form optimizer

\[
t_\star=\min\{1,0.08/(6.5R)\},
\]

and Theorem RE-1. No AST, seed, fixture identifier, or observed decision enters
the formula.

## 3. Postmortem values

| Fixture | Seed | Response energy \(R\) | \(M_8^{RE}\) | RE tail evidence upper | Old tail upper | RE tail upper |
|---|---:|---:|---:|---:|---:|---:|
| AF | 2026081901 | 2.1145039097 | 256.1028224 | 0.00004399813 | 0.03538557 | 0.00166319 |
| AF | 2026081902 | 2.0148885790 | 262.3572641 | 0.00004507263 | 0.02912492 | 0.00139369 |
| AG | 2026081903 | 1.2776095045 | 329.4727827 | 0.00005660299 | 0.00220223 | 0.00012893 |
| AG | 2026081904 | 1.3616218771 | 319.1466945 | 0.00005482898 | 0.00207720 | 0.00011779 |
| AH | 2026081905 | 0.7956031160 | 417.5133564 | 0.00007172825 | 0.00005508 | 0.00000408 |
| AH | 2026081906 | 0.8147336814 | 412.5824739 | 0.00007088113 | 0.00006392 | 0.00000468 |
| AI | 2026081907 | 0.1305245326 | 1030.7955562 | 0.00017708933 | 0.00001795 | 0.00000328 |
| AI | 2026081908 | 0.1294195558 | 1035.1866401 | 0.00017784371 | 0.00001661 | 0.00000305 |

The worst RE posterior-tail upper is `0.0016631860788717072`. Relative to the
old result, the two failed AF bounds are smaller by factors `0.04700` and
`0.04785`.

These numbers show that the theorem targets the observed looseness. They do
not show that a new method has passed. Formal support requires a new
implementation, frozen provenance, and unseen responses.

## 4. Structural successes retained

The semantic quotient remains unchanged:

| Quantity | Value |
|---|---:|
| Cumulative raw AST count through 17 | 5,924,484,194 |
| Size–semantic cells | 31,209 |
| Unique semantic classes | 13,574 |
| Exact core prior mass | 0.9999998282013081 |
| Maximum mass error | 1.11e-16 |

All 64 observation paths in the formal result passed, with total bridge counts
between 13 and 20. Because the RE upper normalizers are no larger than the old
ones, the RE relative-ESS lower bound is algebraically no smaller. This is a
mathematical monotonicity result, not a replacement run.

## 5. Evidence defect to repair before any positive claim

The formal archive did not self-record `source_git_commit`, the confirmatory
runner hash, dependency lock, or fixed interpreter. Source identity was
reconstructed independently from the GitHub freeze commit, and the omission
does not weaken the negative decision. A future positive certificate must be
self-contained and record:

- Git commit and clean-state flag;
- all production/config/runner/test hashes;
- dependency lock hash;
- fixed interpreter path and version;
- platform and NumPy/SciPy versions;
- response materialization timestamp; and
- an explicit parent dependency graph for derived decisions.
