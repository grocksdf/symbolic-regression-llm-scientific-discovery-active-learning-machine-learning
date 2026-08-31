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
- The unique P3I.4 real-development run is frozen as
  `REAL_ADVANTAGE_NOT_DEMONSTRATED`: its shared-H0 protocol and evidence chain
  passed, but no registered efficacy claim passed. P3J addresses the diagnosed
  marginal-transport invariance boundary with normalized class-conditional
  raw-PIT laws. P3J.1 and P3J.2 are correctness/source-composition only: every
  frozen class and likelihood-power target has a strict-prefix state, the same
  pre-reveal calibration factor composes with each target's update, and a
  typed decision authorizes one matching reveal before all four targets advance
  exactly once. Only the nominal `eta=1` target asserts the ordinary joint-law
  Bayes identity; lower powers remain generalized-Bayes sensitivity targets.
  No P3J real execution is authorized until runner composition, cost audit and
  a separately frozen user-only protocol pass.
- P3J.3 reuses each preceding fine quadrature result as the next doubled
  look's exact coarse result under full state/partition identity. The no-data
  ledger reduces worst-case first-query source-class nodes by `32.2581%`, but
  still exposes `101,154,816` cross-class density evaluations at the frozen
  maximum schedule. P3J.3 is therefore a passed correctness/cost audit with
  runner composition explicitly blocked; do not authorize a real run before
  batched, chunked and recoverable scoring passes a separate source Gate.
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
- `P3F.4-CERT.16` passed its identity-bound user execution on source
  `3f7416b3559496464c99c42d4a6078dcd3612d7a`. It uses only the exact slack between
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
- `P3F.4-CERT.17` passed its identity-bound user execution on source
  `1c05dfb81484f6a7eeee79f1290964e4bb964884`. It replaces the invalid requirement
  that one fixed 512-bit precision cover the entire countably-open state space
  with the preregistered threshold-blind schedule `p_r = 512 * 2^r`. Valid
  outward boundary enclosures are intersected before any threshold access, and
  the first envelope whose complete ambiguity bound fits CERT.16's unchanged
  `rho` is certified. Exact rational checks show that both the MH grid floor
  `2 * 2^-256` and the full `N-1`-boundary multinomial grid floor are strictly
  below `rho`; threshold-bit extension is neither needed nor authorized.
  Conditional on actual evaluator widths converging to zero, every finite state
  has a finite eligible round, without claiming one uniform precision or
  runtime ceiling. At this stage operational CERT.13/14 evaluator convergence
  remained unproved and normalization was still quadratic in particle count.
  All evaluator, bit, island and SMC execution flags remained false, and the
  external ideal-bit premise was unchanged.
- `P3F.4-CERT.18` passed its identity-bound user execution on source
  `f2a124baae3915c030d3844549b9231415bb4965`. The actual exact-input CERT.13/14
  function-space builder, collapsed target, exact local/RJ ratio and MH
  probability now accept only the preregistered `512 * 2^r` rounds and compose
  into CERT.17 before threshold access. The Gate checks overlapping, tightening
  512/1024-bit enclosures on the actual path and retains the FLINT inclusion and
  convergence semantics as an explicit third-party numerical premise rather
  than claiming software infallibility. CERT.15 normalization now shares two
  shifted endpoint sums across all particles and excludes each self term,
  replacing the nested pairwise loop by at most `4N` exponential evaluations,
  `O(N)` time and `O(N)` memory. Equal-mass containment, shift invariance and an
  independent 2048-bit point expression are checked deterministically. This
  removes the quadratic normalization bottleneck but does not make the frozen
  `N=212408`, `K=27`, `Q=368876229120` plan experimentally feasible. All
  operational, bit, island and SMC flags remain false.
- `P3F.4-CERT.19` passed its identity-bound user execution on source
  `508c7e1d912751f5ef5d091b0be3580a0e11fb61`. It derives, rather than
  attributes to the published theorem statement, a single-island arbitrary-
  alpha corollary from the free failure parameters in the Marion--Mathews--
  Schmidler appendix. For the CERT.12 values it reduces confirmation to one
  island with `N=69197`, but honestly retains a `4428608000` direct-SMC
  worst-case target-evaluation bound and therefore does not authorize that
  route. The replacement estimator assigns exact positive dyadic tickets to
  every semantic-core atom and one full-support analytic-tail atom, uses
  outward mass bounds to obtain an exact rejection domination constant, and
  verifies on finite rational laws that accepted states are iid from the
  declared posterior. A response-frozen Chernoff proposal cap converts low
  acceptance into whole-procedure abstention without retry, replacement or
  partial output. Fresh fixed-candidate confirmation uses preregistered stages
  and exact rational binomial upper tails at mass `(1-r)/2`, with a Bonferroni
  familywise error budget and no class-count union. The local-mixing SMC theorem
  was audited from its primary source and rejected because its finite
  partition, minimum partition-mass and restricted-kernel premises do not close
  on the countably-open operational target. Actual CERT.14 atom-ball binding,
  ticket/rejection execution, threshold-blind comparison composition, cap
  state machine, production stage freeze and the ideal-bit premise remained
  blocked at that Gate.
- `P3F.4-CERT.20` passed its identity-bound user execution on source
  `745f4deb737d2419b7f691a532dd622a067f00fc`. It binds the complete final-`H0` semantic-
  core/component table of CERT.14 Arb log-marginal balls to CERT.19's exact
  dyadic ticket plan. Outward exponentiation is separated from the exact
  generally non-dyadic rational prior product, so no binary rounding is
  introduced. Exact ticket selection calls the proved semantic-core raw-AST
  lift or the component-times-analytic-tail lift. Rejection probabilities are
  evaluated by the actual CERT.18 evaluator at CERT.17's frozen
  `512 * 2^r` precision values and intersected before each new 256-bit prefix.
  This separately registered lazy-uniform algorithm terminates almost surely
  and returns the exact Bernoulli decision under the explicit FLINT convergence
  and ideal continuous-uniform premises; it does not reinterpret CERT.16's
  unresolved finite-cell event as exact. The external iid ideal-byte law is
  now explicitly accepted and materialized through `secrets.token_bytes`,
  while physical independence remains unproved by source and no deterministic
  PRNG is promoted to an ideal product law. Production freezes `J=17`, 32-bit
  proposal tickets, cap budget `1/100`, an independent 8192-draw empirical-mode
  selection pilot with lexical tie break, and confirmation stages
  `(512,2048,8192,32768)` on a disjoint coordinate domain. A cap miss erases
  every partial state and terminal abstention forbids retry. Operational `H0`,
  real-data, heldout, acquisition, resident-SMC and formal-experiment access
  remain false.
- `P3F.4-CERT.21` passed its identity-bound user execution on source
  `e2787a7f2c6f9303ee32b4ae7b1ae5ec2ee8bd95`. It binds the CERT.20 source,
  CERT.14/17/18 evaluator
  identities and explicit ideal-byte premise into one ordered selection then
  confirmation transaction. Exact proposal audits form a domain-bound SHA-256
  transcript chain. Cap exhaustion and registered Arb, entropy or input
  failures erase all partial accepted marks and return abstention; programming
  assertions and process interrupts are not disguised as scientific
  abstentions. Selection and confirmation byte sources retain disjoint logical
  domains, confirmation stops at the first crossed frozen stage, and an
  overall abstention exposes neither candidate nor selection/confirmation
  state IDs. The canonical terminal ledger is fsynced to staging and published
  by a no-overwrite same-filesystem hard link, carries a complete-payload hash,
  and rejects tampering or rerun overwrite. The operational entry remains
  blocked before provider, entropy and output access. Operational `H0`, system
  entropy, real-data, heldout, acquisition and formal-experiment flags all
  remain false.
- `P3F.4-CERT.22` passed its identity-bound user execution on source
  `851757d1108dc6be46fe46439a85275a5b03ca2a` with the registered decision
  `passed-no-go`. It found that the only frozen target has `feature_count=1` and
  the executable CERT.20/21 source fixture covers only `J=1`, while the
  registered real family comprises CCPP (`d=4`) and two Gas Turbine targets
  (`d=9`) over eight seeds each, hence 24 distinct initial `H0` artifacts.
  Those artifact hashes are intentionally absent because operational response
  access remains closed. An analytic monomial subset gives a strict lower
  bound of 2,350,920 core/component target balls across those histories at
  `J=17`; the full polynomial semantic core can only be larger. No measured
  wall-clock or storage value is guessed. Dimension binding, full-scale source
  construction, operational `H0` identities, and certified time/storage bounds
  are therefore explicit blockers. The Gate may pass only with status
  `passed-no-go`; it reserves a deterministic no-overwrite output identity but
  cannot materialize it or authorize an experiment.
- `P3F.4-CERT.23` is the frozen response-free lazy-source candidate. It removes
  the CERT.22 eager-core blocker at the probability-law level by proposing
  directly from the complete countably-open raw-AST prior times the exact
  registered component prior. Under the existing global likelihood envelope
  `M`, the exact acceptance probability is `L(T,d)/M`; conditioning on
  acceptance therefore gives `p(T,d)L(T,d)/Z` because the proposal prior
  cancels exactly. Each proposal evaluates only its own CERT.18 target ball.
  There is no semantic-shell enumeration, `J`, complete class/component table,
  or dyadic core-atom ticket grid. A response-independent anchor family of
  `one` plus every variable crossed with every component needs only
  `3(d+1)` target balls and gives a positive exact lower bound for the fixed
  no-retry proposal cap. The exact proposal implementation is checked at the
  registered real widths `d=4` and `d=9`. This closes source scalability, not
  operational feasibility: actual H0 binding, the observed response-frozen
  cap audit and wall-clock/storage certification remain blocked, and every
  operational/real/heldout/formal-experiment flag remains false.
- `P3G.1` is the consolidated C1--C3 root repair after the mainline scope
  reassessment. It retains CERT.23 as open-target correctness evidence but does
  not require an operational countably-open sampler for the finite-bank real
  acquisition claim. The real posterior now contains a response-free low-rank
  structure-wise discrepancy sieve with a common spike/kernel nuisance prior.
  Rank-one conjugate updates are checked against batch fitting. One frozen
  runner allocates a `1/100` PIT familywise budget equally across all 24
  seed/target audits; candidate responses stay sealed unless every audit
  passes. A pass automatically triggers the matched-budget comparison using
  joint `(operational class, discrepancy state)` EIG, with random fallback for
  unresolved numerical rankings. A failure emits a terminal calibration NO-GO
  without acquisition. Held-out remains closed and development advantage can
  never be labeled formal efficacy evidence.
- The returned `P3G.1` real run is immutable protocol-valid negative evidence:
  24/24 audits completed, three crossed the `1/2400` per-run boundary, no
  candidate or held-out response was opened, and acquisition count was zero.
  `P3G.2` keeps its splits, seeds, response order, alpha allocation, gate and
  failure policy unchanged. It expands the ordinary generative posterior with
  a response-independent principal-log-variance mixture: one homoscedastic
  state and fixed positive/negative laws on the first three covariate-only
  principal scores, each normalized to unit geometric-mean variance. The
  rank-one update uses `v_i + d_i^T V d_i`; batch and sequential posteriors are
  required to agree. The joint EIG nuisance target now includes the registered
  noise state. These states may learn through marginal likelihood but their
  construction cannot inspect any response. This is a heteroscedastic model
  repair, not a relaxed calibration rule or a seed-specific patch.
- The returned `P3G.2` run is also immutable negative evidence: its
  heteroscedastic states materially reduced several Gas CO e-values but the
  same three tasks rejected. The recorded positive quadratic PIT-basis means,
  sub-`1/12` PIT variances and depleted tail rates diagnose predictive
  overdispersion. `P3G.3` repairs the response-independent source of that
  overdispersion: under initial-development unit-variance design
  preconditioning, a structure with `p` non-intercept terms receives
  coefficient precision `p` for each such term and precision one for the
  intercept. Thus every structure has total non-intercept prior function
  energy one instead of energy growing linearly with representation dimension.
  Batch/rank-one conjugacy, discrepancy/noise mixtures, joint nuisance EIG and
  every real-data Gate boundary remain unchanged.
- `P3G.3` returned 12/24 calibration rejections and is immutable negative
  evidence; fixed function energy is rejected because the posterior transfers
  suppressed signal variation into the common noise scale. `P3G.4` integrates
  function energy instead of selecting another precision. A response-free
  three-point Gauss--Legendre rule discretizes `R^2 ~ Uniform(0,1)` and maps
  nodes to `g=R^2/(1-R^2)`; conditional non-intercept precision is `p/g`.
  Quadrature weights are proper state priors, and structure, discrepancy,
  heteroscedastic noise and energy state form the acquisition nuisance target.
  No dataset, seed, PIT result or response enters construction of this mixture.
- `P3G.4` returned 8/24 rejections and is immutable negative evidence. Six
  were Gas CO. The registered Gas split is an observed year extrapolation
  (2011--12 development, 2013 validation, 2014 acquisition, 2015 held-out),
  while earlier posteriors discarded the year group. `P3G.5` adds one linear
  year nuisance column shared by every Gas structure and standardized only on
  the selected initial-development groups. It is integrated as a common
  coefficient and cannot alter the scientific structure partition. CCPP has
  no group and receives no synthetic regime. The uniform-`R^2` quadrature is
  independently refined from three to five nodes; every response-access and
  calibration boundary remains unchanged.
- `P3G.5` also returned 8/24 rejections and is immutable negative evidence.
  The observed year nuisance reduced the Gas CO geometric-mean maximum-e-value
  ratio to about `0.49` versus P3G.4, but it exchanged resolved failures for
  new failures. CCPP seed `2026080706` persisted. This closes the P3G sequence:
  do not tune another finite coefficient precision, R2 node set, projected-RBF
  rank, noise profile, seed rule, threshold or linear regime against these
  results. A P3H continuation must reconstruct the conditional residual law
  semiparametrically, retain response isolation and the unchanged real-data
  Gate, and pass correctness tests before another user-executed real run.
- `P3H.1` is the response-free correctness Gate for that reconstruction. The
  scientific posterior remains the frozen P3G.5 engine. A separate
  prequential KT dyadic Pólya-tree models only raw base PIT innovations, with
  independent Beta(1/2,1/2) splits and active depth
  `floor(log2(max(history-count,1))/2)`. Current and future responses cannot
  enter an issued forecast; the base scientific update is unchanged. Do not
  call the residual layer a scientific posterior or infer finite-sample
  calibration from its algebraic Gate.
- `P3H.2` is calibration-only. It preserves the 24 runs, response order, PIT
  e-process and `1/2400` per-run allocation. Candidate responses and held-out
  remain sealed even if all 24 runs are nonrejected. A later acquisition Gate
  must integrate the transformed predictive density into the utility; it may
  not reuse the Student-t component EIG implementation unchanged.
- The committed P3H.1 correctness artifact at source `db00f91d` passes all
  eight registered decisions with no real, validation, candidate or held-out
  access. P3H.2 must re-execute this prerequisite before dependency snapshot
  or data loading and must preserve the result in its terminal ledger.
- P3H.2 is bound to the exact P3G.5 runtime dependency hash
  `b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`,
  CPython 3.11.9 and the registered interpreter/DLL/venv binary hashes. The
  environment was intact; the earlier launcher failure inside Codex was a
  sandbox read-boundary issue. Do not reinstall, upgrade, fall back to another
  interpreter or run data if any runtime identity check fails.
- The unique user-executed P3H.2 real calibration-only Gate at source
  `e647cb79` completed 24/24 audits with zero rejection and largest maximum
  e-value `31.856806` against the unchanged `2400` boundary. The immutable
  status is `CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED`; all candidate
  responses, acquisition and held-out remained closed. Treat this only as
  compatibility with the registered PIT betting family. The next admissible
  work is a response-free P3H.3 transformed-law acquisition correctness Gate;
  do not run a real acquisition comparison or reuse Student-t EIG unchanged.
- Resident-SMC integration, independent-island execution, entropy capture,
  product-stream materialization, projector result access, new confirmatory
  materialization, predictive calibration, real data, acquisition, held-out
  access, efficacy claims, and formal paper superiority claims remain blocked.
  CERT.23 has passed; its optional operational continuation would have to bind
  all registered dimensions and H0 identities, evaluate only the frozen linear
  anchor family, and reject before entropy if either proposal cap, wall-clock
  or storage ceilings are infeasible. That continuation is not a prerequisite
  for the finite-bank C1--C3 P3G.1 Gate. A runtime success subset or
  distinct Philox addresses cannot replace the accepted product-law premise.
  Ordinary SciPy CDF values,
  `nextafter`, rounded snapshot arrays, observed-class `other` buckets,
  post-hoc normalization, jitter, regularization and tolerance relaxation are
  not admissible substitutes.
- The user-executed P3H.9 real-development run is an immutable protocol-valid
  negative: 96/96 policy runs and 3072/3072 queries completed with no failures,
  but the terminal assessment was
  `OPERATIONAL_CLASSES_DEGENERATE_NO_CLASS_CLAIM` and both strong-evidence
  flags were false. It must not be rerun. P3I.1 replaces neither evidence nor
  thresholds. It defines operational classes by frozen common-scale posterior
  mean regret, treats the P3H marginal correction as a common invertible PIT
  transport whose class mutual information is invariant, and makes robust
  frozen-class EIG the sole primary utility. The historical P3H.3 KL-projected
  copula and conditional Gaussian-moment EPIG are excluded from the P3I route.
  P3I.1 is response-free correctness only; real, validation, candidate-response,
  held-out, operational and efficacy authorization remain false.
- P3I.2 composes the decision-regret class constructor and target-only robust
  class-EIG into the shared measured-pool runner without changing P3H.9. Under
  its explicit protocol flag, all policies use the same initial-frozen common-
  scale mean-regret partition, while PCPI alone receives the exact P3H
  likelihood-power posterior/residual family. The P3I branch precedes and
  excludes both historical joint scorers; conditional EPIG remains exactly
  zero. Empty representative-safe sets and all execution errors are terminal.
  The first error is fsync-recorded and aborts immediately, with no retry or
  seed replacement. P3I.2 remains source-composition only and is blocked before
  real-data or output access; operational, efficacy and held-out authorization
  remain false.
- P3I.3 changes only the P3I.2 schema, stage and explicit execution
  authorization; its scientific, numerical and assessment configuration is
  otherwise identical. It authorizes exactly one user-executed, supervised,
  held-out-closed real-development run and does not authorize Codex execution.
  The launch must freeze branch/commit/tree/config hash, isolate only historical
  untracked evidence, reject existing output, surface fsync progress and the
  first terminal failure, preserve stderr, stop the child on interruption, and
  restore evidence. A successful process is not sufficient: 96/96 runs, zero
  failures, protocol Gate pass, valid evidence registry, exact manifest identity
  and held-out closure are mandatory. Any protocol-valid negative result is
  immutable and cannot be rerun; this is not independent confirmation.
  The supervised `.ps1` source must remain ASCII-only and pass the actual
  Windows PowerShell 5.1 parser; UTF-8 without BOM plus non-ASCII diagnostics is
  not an admissible launcher format on the user's host.
- The unique P3I.3 run is immutable protocol-invalid evidence. It completed
  96/96 policy runs and 3072/3072 queries with zero execution failures, valid
  evidence and held-out closed, but
  `initial_class_partition_shared_across_policies` was false in all 24
  dataset/seed groups. The three baselines shared one H0 hash and PCPI used a
  second hash because complete-history batch and 16-plus-16 prequential
  conjugate updates accumulated floating sufficient statistics in different
  orders. Do not rerun, overwrite, call it efficacy evidence, round hashes,
  relax the Gate or copy an identity between policies.
- P3I.4 is the only admissible continuation. Under its explicit protocol flag,
  construct exactly one eta-one complete-initial-history posterior and frozen
  decision-regret class target per dataset/seed before the policy loop and
  inject that same typed object into every policy. PCPI must retain all four
  likelihood-power strict-prefix residual states and fail closed unless its
  eta-one sufficient statistics are numerically equivalent under the frozen
  observation-count floating accumulation bound. No data coordinate, budget,
  seed, utility, threshold, numerical schedule, assessment rule or tie break
  changes from P3I.3, and held-out remains closed.
- P3I.4 retains the original canonical CPython 3.11.9 virtual environment used
  for P3H.2 through P3I.3 at
  `D:/01/666/.venv_hypothesis_canonical/Scripts/python.exe`; its imported
  dependency snapshot must equal the frozen
  `b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`.
  Its virtual-environment launcher SHA-256 must equal
  `21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082`
  before the runner checks the data directory, and the manifest must retain it.
  The historical base executable, DLL and `pyvenv.cfg` identities remain
  enforced. A Codex sandbox denial is not a runtime change; do not replace the
  formal environment with an alternate runtime or a Python 3.12 runtime.
  The P3I.4 supervisor must remain ASCII-only, use the encoded-child exit-code
  path, reject null/nonzero exit codes, and preserve the unique-output and
  historical-evidence guards.
