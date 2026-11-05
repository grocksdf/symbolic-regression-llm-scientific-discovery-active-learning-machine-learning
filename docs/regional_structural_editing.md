# Bank-diagnostic-conditioned structural editing

Implementation and synthetic correctness only; no new efficacy evidence.

`region_site_attribution.py` constructs fitting-only region/site features:
local Bank defects, raw subtree behavior, variable dependencies, analytic
parent-output sensitivity, residual association and parent Bank mass/redundancy.
AST categories remain manually specified descriptors; they are not learned
representations. Associations are proposal cues, not causal explanations.
Formula values must exist on the full fitting domain; derivatives need exist
only inside the corresponding region. Unavailable sites are explicitly recorded.

`regional_site_operator_policy.py` fits a shared Normal-Inverse-Gamma linear
working regression to paired Bank log-score differences. Features include
defect/behavior and operator interactions. One posterior draw ranks all legal
site/operator actions. Controls alternate same-site/different-operator and
same-operator/different-site interventions. One averaged fold contrast updates
the model once; invalid, unscored or support-equivalent pairs do not train it.
This adaptive working posterior is not a calibrated scientific posterior.

Transfer is opt-in through an explicitly frozen prior containing training-task
and evidence identities. Feature schema, intervention and utility must match;
the target task must be absent from training. Online state is never shared
implicitly between arms. Transfer capability does not establish transfer efficacy.

The benchmark adapter `scripts/regional_structural_search.py` exposes
`run_regional_site_comparison`. It requires explicit row ancestry, an
ExpansionDesign and lazy independent-role readers. Conditions are random,
local/random, AST-only joint, local joint, attributed joint and shuffled
association joint. The AST-only joint condition differs from the historical
fixed-site baseline. Parent/region/contrast opportunities are frozen as a
Cartesian schedule. All generations freeze before calibration, all calibrations
freeze before admission, and all admissions freeze before reporting.

Search uses standalone candidate predictive components, matching the final
direct expansion family. Fixed materialization slots do not guarantee equal
actual compute: preflight, fitting and candidate-dependent costs may differ.
The adapter performs deterministic symbolic edits without provider calls.
It does not yet route the joint actions into the LLM proposal runtime.
Frontier recycling is disabled to isolate the single-step mechanism.
Historical runners, statistical admission thresholds and old evidence remain
unchanged. No real experiment or sealed data was opened during implementation.
