# Bank-conditioned LLM composition: opt-in development adapter

This change implements a research hypothesis, not a successful optimization or
novelty claim. The previous `bank-location-guided-search-v1` comparison used
symbolic AST edits and zero LLM calls. It cannot establish an LLM contribution.
That implementation and its historical evidence remain unchanged.

## What is being changed

`scripts/bank_guided_llm.py` exposes `run_llm_comparison` under explicit contract
`llm-bank-composition-v1`. A real `ProposalRuntime` generates complete symbolic
equations or parent edits under the existing expanded grammar. The common
frozen Bank inventory provides structural context. The intervention is the
generation information, not a new posterior, relaxed evidence test or changed
reporting metric:

- `bank`: fitting-only whole predictive Bank location diagnostics;
- `residual`: fitting-only MAP-component residual diagnostics;
- `blind`: the same structure inventory, parent and grammar, without diagnostics.

All three modes have the same number of predeclared request slots, candidate
ceiling, new-skeleton quota and provider settings. Each has a fresh proposal
runtime; no KnowledgeRuntime, library retrieval or cross-arm refinement memory
is used. Parent order is fixed without response-based selection. There are no
task-name, seed-specific, formula-specific or reporting-score branches. The
diagnostic is advisory: it does not override the grammar or grant admission.
The common Bank snapshot remains fixed throughout these requests; this is not
a new adaptive multi-cycle feedback algorithm.

## Explicit global/regional composition variant

The same adapter also accepts the separate, opt-in contract
`llm-bank-multiscale-composition-v1`. Its fixed execution order is `blind`,
`global`, `local`, `joint`. It requires an explicit nonempty tuple of distinct
input-feature indices (`diagnostic_axes`) and at least **four** tested mixtures
in `ExpansionDesign`. Invalid axes or an insufficient mixture policy fail
before creating an output directory or opening HTTP. Passing axes to the old
contract is an error; existing three-arm callers are never silently upgraded.

One immutable fitting-only Bank diagnostic snapshot supplies all four arms:

| Arm | Whole-Bank global summary | Input-region summaries | Explicit global/regional links |
| --- | --- | --- | --- |
| `blind` | No | No | No |
| `global` | Yes | No | No |
| `local` | No | Yes | No |
| `joint` | Yes | Yes | Yes |

All arms see the identical structural Bank inventory, fixed parent order,
grammar, allowed actions, new-skeleton quota, candidate ceiling and provider
settings. Each diagnostic uses the **whole predictive mixture**; `local` means
conditioning the summary on a covariate region, not using a local single-model
predictor. Regions are all nonempty median halves for every registered axis.
Axes are not chosen using residuals, admission or reporting outcomes.

Each view reports counts, mean log density, mean signed and absolute location
score, and positive/negative location-score fractions. The location score is
the sensitivity to an infinitesimal common predictive-location shift. It is
not the value of an arbitrary formula edit or a calibrated error certificate.
The joint view adds deterministic regional-minus-global differences and an
opposite-direction flag, without granting additional admission authority.
The global summary in `joint` exactly matches `global`; its region list exactly
matches `local`. Both are projections of the same frozen snapshot.

The global means can already be reconstructed from each complete regional
partition using row-count weights. Consequently, this comparison tests explicit
presentation and integration of global/regional evidence; it does **not** claim
that global and local evidence are independent sources of information. Regions
on different axes overlap and must not be counted as independent observations.
The signed/absolute summaries expose cancellation without selecting favorable
regions or automatically assigning tasks to different algorithms.

`PLAN.json` records axes, scopes and arm execution order. Generation artifacts
and `RESULT.json` record the new contract; prompts use
`bank-multiscale-proposal-feedback-v1`. All four banks freeze before calibration,
all four mixtures before independent admission, and all four decisions before
reporting. The posterior, stacking optimizer, admission test and reporting
statistic are unchanged. Provider/model sampling remains stochastic, and equal
slots do not imply equal tokens or wall time; actual costs remain diagnostics.
This version has no efficacy conclusion and does not launch an experiment.

## Direct predictive components (2026-11-05)

`llm-bank-multiscale-direct-v1` is a separate explicit contract using the same
four-scope adapter. It changes the predictive component family. Legacy callers
retain `nested-bank-v1`: `[p(E0), p(BMA(E0+c1)), ...]`. The new contract uses
`standalone-candidate-v1`: `[p(E0), p(c1), p(c2), ...]`. Each candidate is a
complete expression fitted once on the declared fitting rows under the same
conjugate coefficient/noise model. The protected E0 predictive is identical.
There is no internal competition with E0 before a candidate's density reaches
calibration. The existing simplex optimizer can assign a candidate zero weight.
Independent admission still compares the frozen mixture against E0, and failure
to pass deploys E0. This changes the statistical model family, not the posterior
solver, data likelihood, optimizer, admission threshold or reporting metric.

The reason for this change is structural: evaluating complementary predictors
requires exposing their predictive distributions. If an inner model-selection
average suppresses a candidate, an outer optimizer only sees another copy of E0.
This does not establish that any suppressed candidate is useful. Direct exposure
can also worsen calibration fit generalization; the independent screen remains
necessary and is conditional on its existing null assumptions. Global constant
stacking weights do not implement an input-dependent regional mixture or a
residual-correction model. No claim of solving every local-complementarity
problem follows from this repair.

This follows the distinction between model averaging and predictive stacking in
[Yao et al., Using stacking to average Bayesian predictive distributions](https://arxiv.org/abs/1704.02030).
Our adapter uses a separate calibration split; it does not implement the paper's
PSIS-LOO procedure. Stacking itself is established methodology, not a new
contribution of this change.

The direct contract additionally requires `fit_row_ids` and
`core_generation_row_ids`. Their union must exactly equal `design.training_ids`.
This registers ancestral search/validation access without refitting those rows.
These are caller declarations; the owning matrix must bind them to actual source
row indices and source artifacts. The response-free
`scripts/bank_composition_preflight.py --manifest <freeze.json> --output <new.json>`
checks all historical core dependencies, identities and file hashes. Missing or
failed coordinates reject the entire dependency preflight, without silently
dropping tasks or replacing seeds. Its receipt alone does not authorize execution.

All four arms use the request-bound semantic envelope. The model generates
semantic candidates; the synchronous caller supplies administrative protocol,
round and island metadata. Any conflicting echoed metadata or parent is rejected.
No second format-repair call is enabled. Raw response/prompt hashes and the binding
receipt are retained. This removes an unnecessary metadata-echo dependency, but
does not guarantee valid model output or attest provider-side response freshness.

Calibration and reporting now expose per-component mean log-score differences
and maximum absolute density differences in frozen component order. Reporting
diagnostics are computed only after admission, never fed back to generation.
This permits distinguishing ineffective candidates from weight suppression in a
future run. Historical results and the failed real-development run are unchanged.
No task identity, observed effect, seed or target expression selects this policy.
The current changes were validated using correctness fixtures and mocked HTTP;
no real-data optimization or new provider experiment was performed.

The code does not fine-tune the language model or establish that an explanation
written by the model is scientifically true. It gives the LLM a different,
auditable generation context and preserves independent authority over the
resulting predictor. Adding an LLM or a standard likelihood derivative alone
does not constitute an original research contribution.

## Transaction and statistical boundary

1. Validate explicit code contract, common provider/budget, initial Bank and
   four disjoint roles using `ExpansionDesign`. Only opened development
   fitting arrays are supplied to generation. Row IDs still need binding to
   source indices by an external matrix registration.
2. Generate all modes through the real typed proposal parser, finite formula
   compiler and fitting-domain validity checks. Structurally duplicate
   supports are recorded and excluded. Invalid proposals consume slots;
   they do not receive fabricated zero-valued admission evidence.
3. Durably freeze every candidate Bank before reading calibration labels.
4. Fit the existing `FrozenBankComponents` on fitting data once. Use the
   existing stacking optimizer on calibration; freeze every registered mixture.
5. Apply unchanged `PredictiveExpansionSession` independent admission. The
   caller must register at least three (legacy) or four (multiscale) tested
   mixtures in its evidence policy.
   Freeze all decisions before reading reporting labels.
6. Report all modes, including empty candidate banks, rejected expansions,
   negative gains and failures. A rejected expansion deploys E0 as specified by
   the existing statistical core. This is different from imputing a missing
   proposal's quality as zero.

Fitting-domain validity does not imply validity on unseen domains. A candidate
or core undefined during later density evaluation terminates the run; there
is no silent formula repair or response-dependent candidate deletion. The
general independent-screen guarantee remains conditional on its declared
fixed-law null, not a universal promise of improved future prediction.

## Provider and compute accounting

Actual HTTP transport is used when this function is explicitly invoked with
real provider settings. Tests replace HTTP with synthetic responses and never
read credentials. Importing the adapter does not open a network connection,
dataset, environment file or benchmark.

The configured bounded transport retry policy is fixed and reported. There
are no additional semantic/format repair calls in this adapter. Failed
transport is terminal and preserved, not interpreted as abstention. A 2xx
response that violates the typed output contract is a recorded generation
failure; it does not trigger extra generation or selection of a different seed.

Audits retain request/response hashes, logical calls, attempts, statuses, token
counts when actually supplied, wall time, candidate counts, fitted component
counts and density evaluations. Missing usage remains `null`, not zero cost.
API keys, raw transport error bodies and credential-file contents are not
serialized. Equality of actual tokens/time is not a performance gate. Matching
nominal slots does not guarantee equal computation or prove causal reasoning
superiority; arm execution is sequential in a fixed order.

## Scope and next validation

This is callable experiment infrastructure, not a registered benchmark CLI.
A future runner must freeze task exclusions, source/config/data identities,
provider preflight and credential-file identity, matrix seeds, process timeouts,
uncertainty calculation, failure policy and primary contrast before running.
Provider sampling is not seeded by this adapter's numerical optimizer seed.
No final confirmation is authorized by the code.

The code was developed without opening new task responses or rerunning the
historical eight-task study. Tests cover real typed parsing/compiler/admission
composition with fake HTTP, equal rights, no memory, prompt/reporting isolation,
terminal transport failure, unknown cost, invalid candidates, duplicate supports
and durable freeze ordering. These are correctness checks, not efficacy evidence.
The four-arm tests additionally check exact scope projections, shared search
rights, pre-transport axis validation and reporting-label isolation with real
typed parsing/compiler/admission and mocked HTTP only.

More tasks can improve precision of a later comparison; they cannot establish
an untested generation mechanism or turn a one-task benefit into a general
claim. Any claim of useful Bank-complementary LLM generation still requires
new evidence against ordinary LLM generation and an appropriate non-LLM search
baseline, with resource use reported. We do not claim that this implementation
already distinguishes itself empirically from LLM-SR or influence-guided search.
