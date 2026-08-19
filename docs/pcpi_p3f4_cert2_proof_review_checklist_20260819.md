# P3F.4-CERT.2 proof and implementation review checklist

Status: **REVIEW REQUIRED BEFORE IMPLEMENTATION**

This checklist determines whether the mathematical contract may authorize a
static development implementation. It does not authorize resident SMC.

## A. Target identity

- [ ] The raw countably open grammar and geometric size prior are unchanged.
- [ ] Semantic aggregation retains exact raw-AST multiplicities.
- [ ] The operational predictive-class estimand is unchanged.
- [ ] Coefficient prior mean is exactly zero for every component covered by
      Theorem RE-1.
- [ ] Coefficient and discrepancy precisions are positive definite.
- [ ] Every likelihood power is finite and non-negative.
- [ ] Spike/slab and kernel-state prior weights sum to one.

## B. RE-1 algebra

- [ ] Weighted collapsed marginal likelihood is independently derived from the
      registered NIG target.
- [ ] Matrix determinant lemma gives
      \(|\Lambda|^{1/2}/|\Lambda+X^\top W X|^{1/2}=|V|^{-1/2}\).
- [ ] The residual quadratic is
      \(u^\top V^{-1}u\), with \(u=W^{1/2}y\).
- [ ] The block determinant proof establishes \(|H|\le e^\top H e\).
- [ ] The scalar optimizer handles \(R=0\) explicitly.
- [ ] Fractional powers and second-moment powers above one are covered.
- [ ] The new envelope is never larger than the CERT.1 envelope.
- [ ] Float implementation uses log space and fail-closed finite checks.

## C. Tail and bridge certificates

- [ ] Exact core evidence includes original, unnormalized grammar prior mass.
- [ ] Residual prior mass is exactly \(\rho^J\), or \(\rho^K\) after an exact
      shell escalation.
- [ ] Cutoff escalation schedule and resource ceiling are response-free and
      frozen.
- [ ] Tail decision remains `posterior_tail_upper <= 0.01`.
- [ ] Class-probability/CDF error uses bounded-functional semantics only.
- [ ] Pointwise density is not claimed from the TV tail bound.
- [ ] Every bridge uses RE normalizer uppers in both first and second moments.
- [ ] No forced terminal bridge is permitted.

## D. Practical-kernel mixing

- [ ] Envelope proposal is exactly normalized over semantic core plus raw tail.
- [ ] Tail sampling uses the true conditional geometric size law.
- [ ] Every raw AST sampler returns exact prior probability or an auditable
      proposal probability.
- [ ] Independence-MH acceptance ratio is evaluated exactly.
- [ ] Each local/RJ move has forward and reverse support.
- [ ] Proposal ratios and Jacobians are explicit.
- [ ] Local/RJ target invariance has a unit correctness proof.
- [ ] Macro-kernel order and local-move count are frozen.
- [ ] The TV bound is expressed in macro-sweeps and target-evaluation cost.
- [ ] A tail-root failure produces one root blocker; mixing is marked as a
      dependent blocked decision rather than a duplicate failure.

## E. Evidence and workflow

- [ ] The failed CERT.CF.1 archive remains immutable and NO-GO.
- [ ] AF–AI responses are used only for labelled development postmortem tests.
- [ ] No threshold, seed, fixture, cutoff budget, or kernel frequency is chosen
      to make AF pass.
- [ ] New formal fixtures, coefficients, action grids and seeds are registered
      response-free and disjoint from all seen banks.
- [ ] Freeze source is committed and pushed before response materialization.
- [ ] Commands use
      `D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe` explicitly.
- [ ] Result archives record Git commit, clean state, all hashes, dependency
      lock, interpreter and package versions.
- [ ] Existing output paths fail closed and are never overwritten.

## F. Authorization decision

Static CERT.2 development implementation is authorized only if all items in
A–E are checked and independently reviewed. Even then:

- resident-SMC integration remains blocked;
- unseen confirmatory remains blocked until development passes;
- predictive calibration, real data, acquisition and held-out remain blocked.
