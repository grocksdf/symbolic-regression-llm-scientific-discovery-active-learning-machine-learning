# Bank-informed generation and quality-first search

Date: 2026-11-04. Status: code and synthetic correctness tests only.
No provider, PySR, real benchmark, confirmation or efficacy run was performed.

## Mechanism actually implemented

The old CB-SI scorer already evaluated `mixture(E0, E0+candidate)`. Its donor
heuristic, however, used the best raw formula's residual. These are different
stages; changing admission again cannot repair missing generation information.

`bank_search_feedback.py` computes a dense, signed location sensitivity from
the entire fitted posterior predictive Bank. For

`p_B(y|x) = sum_k pi_k StudentT(y; mu_k(x), scale_k(x), nu_k)`,

the common-location-shift derivative is

`s_i = sum_k r_ik (nu_k+1)(y_i-mu_ik) / (nu_k*scale_ik^2 + (y_i-mu_ik)^2)`.

Here `r_ik` is the component responsibility for that observation. Code uses
the posterior engine's exact design transform, conjugate parameters and
Student-t scale (not its possibly undefined variance). Central differences
against the complete mixture log density verify the derivative. Extreme
residuals use an algebraically equivalent reciprocal expression, without
clipping the prediction, inserting epsilon or changing the fitted law.

The formula is a standard likelihood location derivative. It is a **local
generation surrogate**, not the exact gain of an arbitrary AST edit, a new
statistical theorem, calibrated uncertainty about improvement, or a discovery
certificate. All real candidate gains still use the existing crossfit Bank
scorer and separate final evaluation. In particular, changing an internal
subtree is not mathematically equivalent to shifting every component's location.

## Two integration surfaces

1. Symbolic editor: donor materials are centered and ranked by absolute
   normalized alignment with the location-score vector. The ranked donor is
   actually passed into structural edit requests. Residual and blind controls
   use the same editor, grammar, parent/site coverage and scoring code.
2. Existing Scientist runtime: `install_feedback(runtime, feedback, mode=...)`
   adds search-only regional summaries to the real proposal context. This is
   explicit opt-in. It does not change new-skeleton quota, allowed operations,
   API retry rules or provider budget. One dedicated runtime per condition is
   required; shared/task-local memory reads and writes are forbidden for this
   intervention. Historical Gap prompts and hard-abstention stream policies
   cannot be combined silently with it.

These hooks do not automatically turn old frozen experiment scripts into the
new algorithm. The new adapter contract is `bank-location-guided-search-v1`.
There is no historical result relabeling or claim that an old freeze ran it.

## Controls and information boundary

- **Bank**: complete mixture location score.
- **Residual**: residual of the fitted MAP Bank component, without predictive
  uncertainty weighting. This is a clear ordinary residual baseline, not an
  assertion of bit-identical behavior to the historical raw-expression proxy.
- **Blind**: seeded, response-independent material ordering.

The adapter constructs feedback using fitting rows only. It does not read
search-fold responses to choose donors. Search folds evaluate proposed
candidates and choose the final candidate; repeated use makes these search
scores adaptive, not independent confirmation. Evaluation labels stay behind
a callback until **all** arm selections and `GENERATION_FROZEN.json` are
durably written. Dataset row IDs must be bound to source indices by a future
matrix registration. Opaque caller IDs alone are not proof of no leakage.

Parent and site coverage is deterministic. Least-attempted feasible semantic
operators get priority; this avoids requiring a 45-dimensional online policy
to learn from very sparse rewards. Donors receive round-robin coverage in
priority order. This is an engineering search baseline, not a novel bandit.
Different guidance can change feasibility and the generated candidate pool;
equal nominal opportunities do not mean identical realized work.

The current editor comparison starts from a frozen E0 and frozen archive. It
does not refresh feedback after every discovered candidate, nor does it test
an unrestricted LLM generator. The LLM context hook is available for a future
explicitly registered generator experiment. No claim that the complete
historical feedback loop is now effective follows from this implementation.

## Budget semantics

`SearchBudget` has a mandatory positive finite candidate-attempt ceiling with
no arbitrary 80-slot cap. Default `patience=None` uses the whole declared
ceiling, even after an all-zero cold start. An optional plateau rule can only
consume search gains and must be declared beforehand. Invalid and failed
attempts consume budget; missing scores are not zero observations.

`purpose="quality"` permits different arm ceilings. Its goal is final quality
under declared resources. `purpose="mechanism"` requires common nominal
ceilings/stopping rules. Neither mode makes equal actual calls, tokens or wall
time a performance gate. Planning, materialization, density evaluations,
optimization and time are recorded. Common preparation cost is reported
separately from arm-specific work. These are candidate budgets; a deployment
or formal runner must still register its process timeout and provider limits.

The library has no reporting-based stopping API. Running until a desirable
reporting score appears is not an allowed use of either mode.

## Failure and evidence handling

Each completed attempt is flushed to an exclusive JSONL trace. Invalid edits
and numerical candidate-score failures remain visible. Unexpected scorer
contract/configuration errors terminate the run and preserve a failure record.
There is no hidden resampling, seed replacement, output overwrite or retry.
All arms finish generation before any independent evaluation begins.

The code is a testable mechanism hypothesis. It does not establish improvement,
originality, formula truth, or freedom from overlap with boosting, predictive
stacking or influence-guided symbolic search. Future comparisons must show
whether the new information changes useful structure generation, rather than
merely benefiting from more compute. Quality-first primary evaluation and
smaller controlled mechanism comparisons answer different questions.
