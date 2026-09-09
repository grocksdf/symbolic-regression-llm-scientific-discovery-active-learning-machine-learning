# P3M.6 global--local partial-pooling residual law

P3M.6 is a response-free statistical repair after the immutable P3M.5
real-development result. P3M.5 showed that a candidate-specific RBF residual
law had nearly zero score--realized-gain correlation when the local history was
sparse. P3M.6 addresses that conditional-density-estimation failure directly
without changing the scientific class estimand, posterior family, candidate
pool, budgets, baselines, seeds, validation role, held-out boundary, or risk
assessment.

For the strict prefix of shared raw PIT observations `u_1,...,u_n` and a
candidate action `x`, let `w_i(x)` be the registered RBF weights under the
response-free frozen coordinate transform and bandwidth schedule. The local
effective sample size is

`n_eff(x) = (sum_i w_i(x))^2 / sum_i w_i(x)^2`.

The local weight is fixed before any real response is opened:

`lambda(x) = n_eff(x) / (n_eff(x) + kappa)`, with `kappa = 8`.

The candidate residual law is the normalized common-sieve mixture

`q_x = lambda(x) q_local,x + (1-lambda(x)) q_global`,

where `q_global` is the same strict-prefix KT dyadic Pólya-tree law using unit
weights. Both components use the response-independent depth
`floor(log2(max(n,1))/2)`, so their leaf masses mix directly and remain a
proper density on `[0,1]`. At zero history `n_eff=0` and the law is the global
uniform prior. Numerically stabilized log-weights prevent an all-zero kernel
vector in sparse high-dimensional candidate regions.

The pooling constant and identity are part of the state hash and manifest
method contract. A reveal is still scored under the pre-reveal state and then
advances all four likelihood-power states exactly once. The checkpoint schema,
candidate-action hash, decision-before-oracle ordering, and post-ledger
validation ordering are unchanged.

This stage is no-data correctness and real-development preparation. It does not
claim finite-sample calibration or efficacy; only a new matched-budget user run
can assess whether the repair improves the P3M.5 negative result.
