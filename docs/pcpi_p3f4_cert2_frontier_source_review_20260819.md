# P3F.4-CERT.2 frontier source review

Status: **PRIMARY-SOURCE REVIEW COMPLETE; NO RESIDENT ALGORITHM ADOPTED**

This review asks what current SMC and equivalence-aware MCMC work can safely
change in PCPI after the CERT.CF.1 negative result.  It is not a literature-
driven permission to change the posterior target or tune against AF--AI.

## 1. Direct implications

### Finite-sample adaptive SMC

Marion, Mathews, and Schmidler, *Finite sample bounds for Sequential Monte
Carlo and adaptive path selection using the L2 norm*, arXiv:1807.01346v3
(2025), relates SMC approximation error to adjacent-distribution `L2`
distance and kernel mixing time.  It gives conditions under which relative ESS
can support adaptive path selection and explicitly warns that ordinary data
tempering need not satisfy those conditions.

Primary source: <https://arxiv.org/abs/1807.01346>

PCPI consequence: the CERT.2 bridge remains a **population analytic lower
certificate** built from normalizer bounds.  An empirical CESS number may
choose a path later, but it cannot replace the bound or authorize a forced
terminal step.

### Optimised annealed SMC

Syed, Bouchard-Côté, Chern, and Doucet, *Optimised Annealed Sequential Monte
Carlo Samplers*, arXiv:2408.12057v2 (2025), derives global-barrier objectives
for annealed SMC schedules and supplies open implementations.

Primary source: <https://arxiv.org/abs/2408.12057>

Official code: <https://github.com/alexandrebouchard/sais-gpu>

PCPI consequence: its schedule ideas are a future efficiency candidate only.
They do not establish invariance of a symbolic trans-dimensional kernel, a
posterior-tail certificate, or an inference-to-decision error bound.  No OASMC
component enters CERT.2 static development.

### Standard and waste-free SMC complexity

Le Fay, Chopin, and Vihola, *On the complexity of standard and waste-free SMC
samplers*, arXiv:2604.03352v2 (2026), provides finite-sample results for
expectations and normalizing constants and states explicit limitations where a
spectral-gap or related mixing condition is unavailable.

Primary source: <https://arxiv.org/abs/2604.03352>

Official experiment code: <https://github.com/ylefay/jaxSMC>

Reference implementation: <https://github.com/nchopin/particles>

PCPI consequence: this is the preferred current source for a later comparison
of standard versus waste-free finite-sample ledgers.  It does not rescue the
existing symbolic resident family automatically: PCPI must first provide the
required kernel condition on its actual open symbolic target.

### Identification-aware moves

Kitagawa and Kuang, *Identification-aware Markov Chain Monte Carlo*,
arXiv:2511.12847v1 (2025), shows that moves along observationally equivalent
sets can improve exploration in non-identified posteriors.

Primary source: <https://arxiv.org/abs/2511.12847>

PCPI consequence: equivalence-aware moves are scientifically aligned with the
project, but the paper does not supply the missing PCPI representation map.
The current envelope anchor lives on a semantic-core/raw-tail hybrid space,
while resident moves live on raw ASTs.  A measure-preserving lift or a
lumpability/intertwining proof remains mandatory before composition.

### Independence Metropolis domination

Mengersen and Tweedie, *Rates of convergence of the Hastings and Metropolis
algorithms* (1996), gives the independence-Hastings domination condition used
by the envelope anchor.

Primary source: <https://doi.org/10.1214/aos/1033066201>

PCPI consequence: the envelope minorization is valid only with an exactly
normalized proposal and exact MH correction.  It is not permission to treat
envelope draws as posterior samples.

## 2. Decision for this phase

The response-energy theorem is implemented because it tightens a valid bound
without changing the target and is derived before any new responses.  None of
the frontier samplers above is integrated in this phase.

Future resident integration must compare candidate kernels on the same frozen
target and compute ledger.  At minimum it must expose:

1. common-state-space invariance or an exact lift/intertwining construction;
2. the condition used for its finite-sample mixing/error result;
3. target-evaluation and wall-clock cost per macro-sweep;
4. particle error separated from posterior-tail, path, island, and numerical
   error; and
5. acquisition-decision regret rather than ESS alone.

No source may be used to justify a response-specific rule, dataset branch,
post-hoc regularizer, relaxed threshold, extra seed, or real-data rerun.
