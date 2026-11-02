# CB-SI parent-marginal search repair

Implementation contract: `cbsi-parent-marginal-search-v1`.
This is code and synthetic correctness validation only. Efficacy and original
algorithm priority remain unestablished. Historical v3 results are unchanged.

The policy now learns `G(child; E0) - G(parent; E0)` as its raw reward.
This prevents an edit from inheriting credit for the quality of its parent.
Final candidate selection continues to maximize absolute crossfit bank gain.
Frontier promotion requires positive absolute gain and positive parent-relative
gain. Terminal-depth descendants cannot become new edit parents.

The existing heuristic weighting and priors are retained. Pair observability
is modeled conditional on a scored novel primary; invalid observations cannot
partially mutate policy state. Matched contrasts cancel the parent algebraically
before numerical subtraction. A different equation with identical refittable
parent support is not a novel bank component.

Evaluation opens lazily after an exact candidate selection is sealed. Search
cannot resume after sealing. Exact-expression caches avoid repeated densities
and stacking fits within the same fixed scorer; aliases are not merged merely
because they share support. Counters include baseline and candidate density
attempts, optimizer fits and cache hits. Arm timing includes scorer setup.

Old freezes are rejected before data access. The preparation command creates
an unauthorized draft under the new implementation contract; no experiment is
authorized or executed by this repair. Existing v3 finalizer remains a historical
checker and is not the finalizer for the new protocol. A new execution protocol
and evidence checker must be reviewed before any efficacy run.

Research hypothesis: matched, parent-relative bank credit may guide useful
structural edits. Standard Thompson sampling is not a novelty claim. The search
folds remain adaptively reused and cannot establish independent efficacy.

References informing boundaries:
- Russo et al., A Tutorial on Thompson Sampling (2018):
  https://arxiv.org/abs/1707.02038
- Dwork et al., Generalization in Adaptive Data Analysis and Holdout Reuse (2015):
  https://arxiv.org/abs/1506.02629
