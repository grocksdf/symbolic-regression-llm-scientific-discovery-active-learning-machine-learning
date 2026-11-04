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
   existing stacking optimizer on calibration; freeze all three mixtures.
5. Apply unchanged `PredictiveExpansionSession` independent admission. The
   caller must register at least three tested mixtures in its evidence policy.
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

More tasks can improve precision of a later comparison; they cannot establish
an untested generation mechanism or turn a one-task benefit into a general
claim. Any claim of useful Bank-complementary LLM generation still requires
new evidence against ordinary LLM generation and an appropriate non-LLM search
baseline, with resource use reported. We do not claim that this implementation
already distinguishes itself empirically from LLM-SR or influence-guided search.
