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
- `P3F.4-CERT.2` authorizes only a response-energy, dependency-aware static
  development certification layer after proof-review closure.
- Resident-SMC integration, new confirmatory materialization, predictive
  calibration, real data, acquisition, held-out access, efficacy claims, and
  formal paper superiority claims remain blocked until their preceding Gates
  pass.
