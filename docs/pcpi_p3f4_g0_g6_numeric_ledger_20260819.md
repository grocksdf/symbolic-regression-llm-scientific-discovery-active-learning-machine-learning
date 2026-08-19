# P3F.4 G0--G6 numerical proof ledger

This ledger records the inputs and derived values used by
pcpi_p3f4_g0_g6_proof_package_20260819.md. It is a static mathematical audit,
not an inference run.

## Frozen inputs

| Quantity | Value | Source |
|---|---:|---|
| Source commit | ffe7239955d9083a7ad6ef878c3213c602027aad | GitHub main baseline |
| VR.8 archive SHA-256 | 16bb9ab7e007fd98614ebb3152f929501db75655882958a1c256770ada6d235a | Uploaded frozen archive |
| Response count | 8 | VR.8 matched budget |
| NIG shape \(a_0\) | 3.0 | P3F.2 target config |
| NIG scale \(b_0\) | 0.08 | P3F.2 target config |
| Grammar continuation \(\rho\) | 0.4 | P3F.2 target config |
| Reference cutoff \(L\) | 3 | P3F.2/P3F.3 configs |
| Slice prior mass | 0.936 | \(1-\rho^L\) |
| Tail prior mass | 0.064 | \(\rho^L\) |
| Collapsed size-three states | 42 | 14 ASTs times 3 discrepancy states |
| Relative-ESS floor | 0.8 | Frozen P3F.3 path control |

## Uniform marginal-likelihood envelope

\[
\log M_n=
-\frac n2\log(2\pi)
+\log\Gamma(a_0+n/2)-\log\Gamma(a_0)
-\frac n2\log b_0.
\]

At \(n=8\):

\[
\log M_8=8.637510343045799,
\qquad
M_8=5639.272478769489,
\]

and

\[
U_{>3}=M_8\rho^3=360.91343864124735.
\]

## Evidence and posterior-tail conversion

The archive reports the conditional size-three log evidence. The full-target
slice contribution is

\[
Z_3=(1-\rho^3)\exp(\log Z_3^{\rm cond}).
\]

| Fixture | Conditional log evidence | \(Z_3\) | \(\bar\tau_3\) |
|---|---:|---:|---:|
| AC | -1.8529273967482702 | 0.1467437810166268 | 0.9995935752870001 |
| AD | -1.0027040461015804 | 0.3434053165299543 | 0.9990494150567010 |
| AE | 3.4595117830413150 | 29.76615412844351 | 0.9238092936530960 |

## Raw-AST enumeration growth

| Maximum size | ASTs at size | Cumulative ASTs |
|---|---:|---:|
| 10 | 126,626 | 165,588 |
| 11 | 556,778 | 722,366 |
| 12 | 2,446,138 | 3,168,504 |
| 13 | 10,930,130 | 14,098,634 |
| 14 | 49,027,890 | 63,126,524 |
| 15 | 222,074,010 | 285,200,534 |
| 16 | 1,010,597,130 | 1,295,797,664 |
| 17 | 4,628,686,530 | 5,924,484,194 |
| 20 | 456,507,632,730 | 582,129,971,144 |

Using only \(Z_3\) as the evidence lower bound, the smallest \(L\) sufficient
for \(\bar\tau_L\le0.01\) is 17 for AC, 16 for AD, and 11 for AE.

## Open independence-kernel minorization

\[
\underline\epsilon_T=Z_3/M_8,
\qquad
m_{0.01}=
\left\lceil
\frac{\log(0.01)}
{\log(1-\underline\epsilon_T)}
\right\rceil.
\]

| Fixture | \(\underline\epsilon_T\) | \(m_{0.01}\) |
|---|---:|---:|
| AC | 2.6021757517e-5 | 176,972 |
| AD | 6.0895322548e-5 | 75,623 |
| AE | 5.2783677754e-3 | 871 |

## Analytic path certificate

For every bridge \(\beta<\beta'\),

\[
\underline r_t(\beta,\beta')
=
\frac{Z_{L,t,\beta'}^2}
{\overline Z_{t,\beta}
\overline Z_{t,2\beta'-\beta}}.
\]

Exact enumeration of the 42-state slice and the analytic tail envelope gives:

| Fixture | Minimum on frozen beta grid | Worst bridge | Retained-evidence fraction at observation 8, \(\beta=0.75\) |
|---|---:|---|---:|
| AC | 4.5000710457e-8 | 8, 0.75 to 1 | 0.0002779638483 |
| AD | 2.1835498372e-7 | 8, 0.75 to 1 | 0.0006175483735 |
| AE | 0.0096027393768 | 8, 0.75 to 1 | 0.1072059539282 |

The earlier beta-zero path blockers are:

| Fixture | First blocked observation | \(Z_{L,t,0}/\overline Z_{t,0}\) |
|---|---:|---:|
| AC | 2 | 0.339598532 |
| AD | 2 | 0.324976838 |
| AE | 3 | 0.739431171 |

Since these fractions are below 0.8, the analytic certificate cannot authorize
any positive next bridge at those observations.

Across all 96 frozen fixture/observation/beta cells:

- maximum log-convexity log violation: 0.0;
- maximum analytic-lower-bound minus exact-slice relative ESS: 0.0.

## Integrity boundary

The ledger used only:

- frozen target hyperparameters and configs;
- the archived exact finite-slice evidence;
- exact enumeration of the already registered 42-state slice; and
- analytic inequalities stated in the proof package.

It did not run SMC, choose a seed, alter a threshold, access real data, open
held-out state, or write inference code.
