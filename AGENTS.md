# PCPI AISTATS research contract

Read `.cursor/rules/aistats-research-guardrails.mdc` and the relevant decision
record before changing research code, protocols, evidence, or paper claims.
The rules below are the repository-safe summary of the user's canonical
long-horizon context and remain binding even when a local task omits them.

## Scientific objective

- PCPI studies sequential Bayesian discrimination among frozen operational
  predictive-equivalence classes of symbolic laws.
- Keep C1--C3 primary: a frozen estimand and coherent posterior; target-correct
  SMC; and acquisition that reduces the registered scientific Bayes risk.
- Treat LLMs, motifs, discrepancy variants, model expansion, and VED as
  optional extensions.  They may improve proposals or efficiency but may not
  silently change the posterior target, estimand, utility, or evidence role.
- Aim for an AISTATS-quality conditional contribution, not a universal
  superiority claim.

## Evidence and negative-result discipline

- Gates are serial: exact-reference correctness -> target-correct SMC ->
  acquisition correctness -> audited matched-budget real development -> one
  sealed confirmation -> submission claims.
- A failed Gate is a permanent result for that frozen protocol.  Do not rerun
  with favorable seeds, change thresholds or budgets, delete failures, or
  relabel development evidence as confirmation.
- Never run real or held-out experiments merely to search for a positive
  result.  Negative or uncertified results first trigger a task-independent
  root-cause analysis of the estimand, probabilistic model, posterior
  approximation, calibration, uncertainty propagation, or decision objective.
- Repairs must follow an explicit probability/decision-theory contract and,
  where relevant, current primary papers plus official source implementations.
  Validate them on exact-reference or response-free correctness fixtures
  before freezing any new real protocol.
- Never improve reported results through answer, dataset, target, formula, or
  direction hard-coding; post-hoc regularizers or rules; seed selection;
  threshold relaxation; extra compute; hidden retries; or privileged LLM,
  memory, engine, validation, or held-out information.
- Preserve all null and negative evidence and make claim boundaries explicit.
  Correctness, calibration, protocol validity, efficacy, and discovery are
  separate claims.

## Data and experiment boundaries

- Synthetic or controlled fixtures support correctness/calibration diagnostics
  only.  They cannot support superiority, scientific discovery, or real-data
  efficacy.
- Preserve development, validation, candidate-pool, and sealed-test roles.
  Do not inspect sealed arrays, paths, shapes, ranges, metadata, summaries, or
  outcomes before the registered one-time confirmation.
- Codex may run static, unit, algebraic, and non-efficacy smoke checks.  Formal
  experiments and user compute budgets are run only by the user with explicit
  commands.  A bad development result does not authorize a real run.
- Real comparisons require the same initial data, measured-label budget,
  posterior engine, particle/SMC controls, candidates, compute ceiling, seeds,
  splits, failure policy, and LLM-call policy, with paired uncertainty and full
  failure accounting.  Acquisition-off means matched-budget random
  acquisition, never no acquisition.

## Inference and acquisition boundaries

- Call an object a Bayesian posterior only when code targets the declared
  normalized prior-likelihood model.  Label power posteriors, robust envelopes,
  generalized Bayes targets, surrogates, and bridges honestly.
- Every SMC/MCMC kernel must prove target invariance, complete forward/reverse
  support, proposal ratios, and any Jacobian.  ESS/CESS choose a numerical path;
  they do not replace a posterior-functional or decision-error guarantee.
- Semantic quotients and hybrid state representations require an explicit
  measure-preserving lift/lumpability proof before composition with raw-AST
  local/RJ kernels.
- Call a score exact EIG only after an independent exact-reference check.
  Unresolved numerical ranking must fail closed or use the registered interval
  decision; it must not switch silently to another scientific objective.

## Repository and provenance

- GitHub `main` is canonical.  Start every task by checking HEAD, branch,
  worktree, remote, submodules, manifests, and relevant evidence identities.
- Preserve one production implementation.  Historical frozen runners may stay
  isolated for reproducibility, but do not create `old/new/final/fixed/v2`-
  style parallel production mechanisms.
- Do not overwrite result directories.  Record source commit/tree, clean state,
  configuration and dependency hashes, fixed interpreter, seeds, budgets,
  hardware, package/provider identities, fallbacks, diagnostics, failures,
  held-out flags, and claim boundary in the evidence lineage.
- On the user's Windows host, commands must explicitly use
  `D:\01\666\hypothesis_mvp` and
  `D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe`; do not depend on
  the current directory, globs, or unvalidated environment variables.

## Current authorization boundary

- `P3F.4-CERT.CF.1` remains a permanent confirmatory NO-GO.
- `P3F.4-CERT.2` supplies the response-energy, dependency-aware static
  certification layer; `CERT.3` supplies the exact semantic-core/raw-AST lift;
  and `CERT.4` supplies a standalone complete raw `(T,d)` envelope anchor.
- The CERT.4 resident audit recorded three composition blockers: the open
  resident sampler had an `int64` shell ceiling, raw-expression floating
  evaluation was not exactly class-constant, and no resident raw-AST local/RJ
  proposal was implemented. Finite independence-proposal formulas pass only
  on their declared finite support. Preserve these findings as the historical
  baseline for later source repairs.
- The first Windows CERT.4 response-free execution at `8b18f74f` is retained
  as a failed check attempt: three tests asserted bit-exact floating
  normalization although the implementation had already registered and
  enforced a `2e-12` log-mass identity tolerance. CERT.4-R1 may align only
  those assertions with that pre-existing tolerance; it may not change the
  target, proposal, envelope, resident decision, or scientific Gate.
- `P3F.4-CERT.5` supplies a standalone exact involutive raw-state local/RJ
  proposal. It chooses a raw-AST address uniformly, regenerates a complete
  countably-open subtree and exact component, retains the discarded values as
  the reverse auxiliary state, and evaluates target mass only through the
  exact polynomial key. Root replacement gives complete bidirectional support;
  exact pathwise proposal masses and a discrete unit Jacobian give target
  invariance. This construction is not imported by resident SMC and does not
  authorize resident integration.
- `P3F.4-CERT.6` repairs the first two resident source boundaries without
  running the engine: fixed-shell and complete open-prior raw-AST draws use
  arbitrary-precision exact tickets, while resident base designs, projected
  discrepancy bases, and caches are keyed only by the exact polynomial class.
  Its standalone common-target adapter evaluates CERT.5 proposal endpoints
  through that semantic interface and retains the exact proposal ratio. The
  adapter deliberately does not import `_rejuvenate` or call
  `ScalableOpenTargetSMC.run`; resident integration remains unauthorized until
  a separate source-composition Gate proves the actual rejuvenation path.
- `P3F.4-CERT.7` imports the proved proposal and common-target adapter into an
  actual `raw-state-local-rj` branch of resident `_rejuvenate`. The branch is
  restricted to the complete open support (`maximum_nodes=None`) and the
  terminal-only population mode. Resident endpoint particles must match the
  exact raw/component proposal endpoints, semantic aliases must have identical
  collapsed log marginals, and resident versus adapter target log masses must
  agree before the adapter-owned proposal ratio and acceptance value can be
  used. A finite censored matrix built through this resident endpoint helper is
  stochastic, reversible, and invariant. The corresponding `run()` path is
  hard-blocked before data validation or particle sampling, so this is a
  source-composition result rather than resident-SMC execution authorization.
- `P3F.4-CERT.8` binds the response-energy analytic normalizer certificate,
  exact rational beta grid, incremental potential, systematic resampling and
  CERT.7 rejuvenation to one immutable resident Feynman--Kac target identity.
  It chooses the largest grid step whose population relative-ESS lower bound
  is at least `0.8`; empirical CESS is diagnostic only, a forced terminal step
  is forbidden, and only the response prefix through the current observation
  may enter the path. Exact finite resampling is unbiased, the resample-move
  composition is invariant, and the actual censored local/RJ matrix is
  irreducible, aperiodic and has a positive finite spectral gap. The `run()`
  guard remains before target, response and particle access, so CERT.8 is
  still a response-free source-composition result rather than execution
  authorization.
- `P3F.4-CERT.9` adds a decision-derived finite-particle and independent-island
  error budget without running the resident engine. The registered 2025
  fixed-path `L2` theorem requires conditionally independent multinomial
  offspring at every bridge, so CERT.8 systematic resampling remains an
  unbiased composition fact but is explicitly rejected as a finite-`N`
  theorem shortcut. The certifiable resident branch instead resamples
  multinomially after every bridge and mixes exact-prior independence MH with
  the proved local/RJ kernel at a frozen probability. The response-energy
  core evidence and global likelihood envelope give a countably-open
  minorization for that mixture. Actual resident target and proposal ratios use
  arbitrary-precision-prior-derived log masses, so floating underflow cannot
  remove a legal raw AST. Every bridge must satisfy its derived mixing depth
  before any particle is sampled. Per-island finite-`N` success is
  amplified only across independent islands by a componentwise median with an
  exact binomial-tail/union budget. Particles within an island remain dependent
  and are never counted as replicates. Class-coordinate error is converted to
  class total variation and the frozen MAP-regret budget rather than chosen
  from observed performance.
- `P3F.4-CERT.10` adds the response-free independent-island executor and
  componentwise-median source composition. It binds the complete CERT.8
  Feynman--Kac plan, CERT.9 finite-`N` plan, exact particle configuration,
  support-extension-invariant operational estimand, ordered class identities,
  projector identity and product-law coordinates. Distinct integer seeds,
  spawned pseudorandom streams and distinct generator objects are not promoted
  to a mathematical independence proof; an external product-randomness premise
  remains explicit and its production implementation is unauthorized. Shared
  generator, aliased or exactly duplicated bit-generator state, and crossed
  coordinates are rejected. Every
  coordinate and island failure is preserved in one batch, with no retry,
  replacement or partial output. Componentwise medians are unnormalized MAP
  decision scores: they are never normalized, projected onto a simplex or
  exposed as posterior probabilities. The direct MAP regret bound is
  `2 * epsilon`, and the exact binomial-tail union budget remains bound to all
  operational classes. The executor guard precedes product-source, projector,
  response, engine and particle access, so CERT.10 is not an execution result.
- `P3F.4-CERT.11` implements, without executing, an auditable direct-key
  product-source and the complete implicit operational-class map. Exactly one
  external 128-bit key is bound to every ordered island coordinate and enters
  `Philox(key=..., counter=0)` directly; no root key, integer-seed argument,
  `SeedSequence.spawn`, jump, retry, replacement or favourable-key selection
  is used. The operating-system entropy tuple remains an explicit external
  product-law premise and is not claimed to be proved by source inspection.
  The projector freezes `H0`, the standardizer, action/threshold grids,
  `B=32`, `K_B=6` and the grid-only claim, then gives all `6^(7*|A0|)` classes
  reversible base-six ranks without enumerating them. Exact rational CDF
  intervals that intersect multiple bins retain sparse boundary-uncertain
  mass and exact class lower/upper bounds; they are never rounded to the
  nearest bin. The CERT.10 fixed-vector adapter fails on any uncertainty or
  occupied unregistered class and never adds `other` or renormalizes. Four
  additional guards precede entropy, coordinate, result, particle and oracle
  access, while all three CERT.10 guards remain false.
- `P3F.4-CERT.12` implements the response-free rigorous special-function
  kernel and replaces the intractable all-class union budget for MAP decisions
  with a selection/confirmation product split. The numerical kernel accepts
  only exact dyadic parameter balls, pins `python-flint==0.8.0` and 256-bit Arb
  arithmetic, evaluates the Student-t CDF through the regularized incomplete
  beta function, and exports exact outward binary endpoints. Analytic Cauchy
  identities are checked, but rounded resident posterior arrays are explicitly
  rejected as rigorous inputs: the full `H0` state-to-parameter-ball provider,
  including the structurewise RBF discrepancy design, remains unproved and
  blocked. For the MAP decision, one selection island may name an arbitrary
  candidate; nine fresh independent confirmation islands evaluate only that
  now-fixed class indicator. Conditional on selection, the CERT.9 fixed-
  functional theorem therefore applies without a union bound over
  `6^(7*|A0|)` classes. With error `r/2`, confirmation median at least `1/2`
  proves candidate mass at least `(1-r)/2` and MAP regret at most `r`; otherwise
  the procedure abstains. Selection and confirmation coordinates are disjoint,
  no island is reused, and there is no retry, seed selection, class-space
  enumeration, result-derived threshold, normalization or posterior-vector
  claim. The external product-entropy premise remains explicit.
- `P3F.4-CERT.13` reconstructs complete predictive Student-t parameter balls
  from an exact frozen `H0` identity rather than resident NumPy snapshots. It
  evaluates every polynomial semantic key over the exact registered binary
  domain, chooses varying standardizer coordinates by exact equality, builds
  the RBF kernel in 512-bit Arb, and replaces the floating `eigh`/SVD factor
  with the factorisation-free conditional covariance
  `K - K g (g^T K g)^-1 g^T K`. The zero polynomial receives the exact vacuous
  constraint. A validated `arb_mat.solve(..., algorithm="precond")` then gives
  the conjugate Gaussian/NIG location, scale-squared and degrees-of-freedom
  balls for every frozen action/threshold coordinate. No approximate solve,
  matrix inverse, jitter, regularizer, tolerance rank, precision retry or
  rounded posterior input is admitted. CERT.13 also repairs the CERT.12
  special-function endpoint at zero by exact beta-domain intersection and a
  complementary incomplete-beta identity, and repairs the majority-mass
  regret certificate to use the nonnegative upper bound
  `max(0, 1 - 2*lower)`. A candidate-only sparse projector propagates exact
  lower/upper mass for the selection-fixed class without materializing a full
  `6^d` vector; its lower bound composes with confirmation while all uncertain
  compatible mass remains only in the upper bound. Only the standalone
  algebraic constructor is authorized. Operational `H0`, CDF/projector result,
  island and SMC access remain blocked.
- `P3F.4-CERT.14` replaces authority for the resident floating factor-basis
  collapsed target with the same 512-bit Arb function-space covariance builder
  used by CERT.13. For an exact rational bridge weight matrix `W`, it evaluates
  the symmetric system `I + sqrt(W) P sqrt(W)`, its determinant and validated
  `precond` solve, and the Gaussian/NIG weighted collapsed log marginal. Only
  the response prefix through the current bridge coordinate may be accessed;
  changing a future frozen response leaves the earlier ball unchanged. Bridge
  potentials are outward differences of two balls carrying one immutable
  target identity. Local/RJ acceptance combines those balls with exact raw-AST,
  component and forward/reverse auxiliary fractions and a unit Jacobian; an
  unresolved interval is not converted to a decision. Exact finite rational MH
  matrices verify detailed balance and target invariance, and the sparse
  fixed-candidate adapter carries the same target hash without normalization or
  class-vector materialization. The resident float target branch now fails
  closed before use, and all operational target/result/island/SMC flags remain
  false. No inverse, approximate Arb, midpoint promotion, jitter, regularizer,
  tolerance rank or result-dependent precision retry is admitted.
- `P3F.4-CERT.14-R1` repairs only the response-free syntax-population identity.
  Syntax compilation is now over the exact Git-tracked `*.py` source set, not
  every filesystem Python file. Ignored environments, build products and other
  untracked local artifacts therefore cannot change the Gate count, while any
  tracked invalid source still fails. One deterministic fixture proves both
  directions. The common target, all execution guards, experiment prohibitions,
  numerical precision and scientific claim boundary are unchanged.
- `P3F.4-CERT.15` passed its identity-bound user execution on repaired source
  `7f8f961900c8f1cd739bb84d23c33c57f848640e`. It normalizes CERT.14 log-mass
  balls with monotone outward Arb ratios and interprets each fixed 256-bit
  string as the complete half-open dyadic cell of ideal uniforms sharing that
  prefix. Multinomial inverse-CDF or MH output is returned only when the whole
  cell has one decision for every target represented by the probability ball;
  otherwise the complete operation fails with an explicit unresolved-
  probability upper bound. No midpoint, discrete left-endpoint surrogate,
  modulo reduction, adaptive bit extension, precision retry, replacement or
  partial ancestor vector is admitted. The small dyadic law check is exact
  deterministic enumeration, not a simulated experiment. Only standalone
  comparison composition is a candidate authorization; product-bit
  materialization, resident resampling/MH, islands and SMC remain false. The
  passing response-free Gate supports these comparison semantics only and does
  not by itself compose their failure probability with the finite-`N` theorem.
- `P3F.4-CERT.15-R1` repairs the inherited full-suite baseline before any
  CERT.16 work. The historical confirmatory `certification.py` is restored to
  its byte-frozen SHA-256 and resident prefix padding is isolated in the
  CERT.8 adapter. The canonical editable environment identity is synchronized
  to project version `1.4.0`, the P3F.3 validation test names every registered
  resampling schedule, and the final-source audit retains its 80/100-line
  limits while accepting legacy long functions only under complete normalized
  source-segment SHA-256 identities. New or edited over-limit functions still
  fail closed. The inherited five failures are therefore repaired without
  changing an experiment, response, numerical tolerance or scientific result.
- `P3F.4-CERT.16` is the frozen response-free conditional integration candidate
  awaiting identity-bound user execution. It uses only the exact slack between
  CERT.9's independent-island bound and the registered simultaneous failure
  probability, and freezes one equal per-comparison allocation over the full
  `K*N*S*(1+M)` coordinate space. The implicit rank is a complete bijection;
  each coordinate binds its CERT.11 island-stream hash, key commitment and a
  domain-separated 256-bit Philox counter block. An over-budget local bound is
  rejected before bit access and any unresolved comparison aborts the complete
  island batch without retry or partial output. The Gate proves only the
  conditional algebraic identity `B_N + Q*rho = alpha`: a runtime precheck does
  not bound the probability of entering an over-budget state. A uniform
  response-free comparison envelope over all reachable states remains false.
  Philox address uniqueness is proved, but PRNG output is not promoted to
  mathematical ideal-bit independence. All bit, island and SMC execution flags
  remain false.
- Resident-SMC integration, independent-island execution, entropy capture,
  product-stream materialization, projector result access, new confirmatory
  materialization, predictive calibration, real data, acquisition, held-out
  access, efficacy claims, and formal paper superiority claims remain blocked.
  The next admissible action is the one-time response-free CERT.16 Gate on its
  frozen source identity. After a pass, the next theorem phase must establish a
  response-independent uniform ambiguity envelope over every reachable
  pre-threshold state and state an explicitly accepted ideal-bit/product-law
  premise; a runtime success subset or distinct Philox addresses cannot replace
  either proof. Ordinary SciPy CDF values,
  `nextafter`, rounded snapshot arrays, observed-class `other` buckets,
  post-hoc normalization, jitter, regularization and tolerance relaxation are
  not admissible substitutes.
