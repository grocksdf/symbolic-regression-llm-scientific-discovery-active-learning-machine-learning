# Comparator-aware structural search

Status: implementation and synthetic correctness checks only. No efficacy
experiment or sealed data access is included. Contract:
`cbsi-comparator-aware-search-v1`. Historical results keep their original source.

## Source finding and focused hypothesis

The scalar credit learner records which operator produced the primary candidate
but does not learn the identity of its changing comparator. Winning against a
weak control and losing against a strong control therefore have different
interpretations that a primary-only reward mean cannot express. The previous
weighted sum of raw and relative credit also treats correlated evidence as
separate working models.

The new experimental arm retains the paired comparison explicitly:

    d = mean_k[G(primary)_k - G(control)_k]
    x = phi(primary, context) - phi(control, context)
    d | theta, sigma^2 ~ Normal(x theta, sigma^2)

Features consist of global operator indicators plus context-specific operator
indicators. Contexts are fixed AST site kinds and root/internal location.
A joint Normal-Inverse-Gamma working model updates once per valid pair. One
joint Thompson draw ranks the admissible operators. There is no feasibility
multiplier or raw-plus-credit coefficient. Differences identify relative skill,
not an absolute reward level; ranking is invariant to a common action offset.
The conjugate Bayesian regression and Thompson sampling are standard components,
not claimed as original algorithms. The iid linear noise assumption is a working
approximation: adaptively reused search folds do not confer calibrated scientific
posterior probabilities or causal operator effects.

Both candidates are scored independently when executable and novel. A missing
primary does not erase a valid control. Both may enter the search candidate pool;
neither acquires posterior authority from the operator learner. The final single
candidate is chosen by the same absolute crossfit bank gain and is sealed before
independent development evaluation. No threshold or existing admission rule is
relaxed.

## Fair comparison and limitations

`bank_greedy_paired` receives the same static opportunities, semantic preflight,
two reserved candidate slots, both-candidate retention, cache and final selector.
Its raw-gain greedy operator learner receives both observed rewards. Thus the
primary comparison controls for keeping the previously discarded controls.
Legacy arms remain diagnostic ablations, not equal-access comparators.

Actual density calls, optimization work and wall time must be reported. Equal
reserved slots and identical maximum work bounds do not mean equal realized CPU.
No descendant frontier is used in either new arm. The protected E0, fit roles,
search folds and final evaluation split retain their existing roles. Missing or
invalid pairs never create zero or favorable reward observations.

The builder produces an unauthorized new draft; no old execution authorization
transfers. The proposed development primary is cbsi_paired minus
bank_greedy_paired with a task-paired 95% bootstrap lower endpoint above zero,
plus positive absolute cbsi_paired mean. Other contrasts are diagnostic. Reusing
historical tasks remains retrospective development, even with a new source freeze.

## Position against consulted primary source implementations

- [LLM-SR buffer](https://github.com/deep-symbolic-mathematics/LLM-SR/blob/41c212312df6c16d936c9cb395356a62774c47e3/llmsr/buffer.py)
  organizes score-driven program evolution into islands. Score-guided generation
  alone is already established.
- [IGSR](https://github.com/DrShushen/IGSR/blob/44af1f48ce0881987cc996657b5e5c60b8661e4f/README.md)
  uses validation-error term removal influence in propose/prune search. Giving
  a generator contribution feedback alone is already established.
- [auto-sklearn ensemble selection](https://github.com/automl/auto-sklearn/blob/development/autosklearn/ensembles/ensemble_selection.py)
  evaluates candidates conditional on the ensemble. Bank-conditional value
  alone is not sufficient as a novelty claim.
- [Contextual Thompson sampling](https://proceedings.mlr.press/v28/agrawal13.html)
  and [contextual dueling bandits](https://proceedings.mlr.press/v235/li24co.html)
  establish related selection approaches. We use continuous paired score
  differences, not the latter paper's binary-preference algorithm or theorem.

The remaining research hypothesis is whether matched, comparator-aware structural
edit evidence improves bank-value search under equal opportunities, while keeping
generation authority separate from predictive evidence. This source review does
not prove priority, non-overlap with all literature, or empirical superiority.
