# Expanded formula discovery: development NO-GO audit and repair boundary

The 36 registered development rows retain their original `passed=false` result. Ground truth was opened only after admission freeze. Test/OOD and untouched confirmation remain closed. This note analyzes the attached, already opened development handoff; it does not rescore historical rows.

## What failed

| Observed evidence | Consequence |
| --- | --- |
| 648 HTTP 429 and 13 HTTP 200 in 661 registered provider requests across 36 child ledgers; 34 children had zero 2xx | Most runs supplied no successful model response. A transport failure cannot count as evidence that the LLM cannot propose useful formulae. The exact cause of 429 remains unverified without a provider quota/error audit. |
| The three admission banks are identical in 35 of 36 rows | This run has essentially no variation with which to identify an admission policy effect. |
| The sent user prompt was always 65,536 UTF-8 bytes | The child padded prompts with spaces to the registered ceiling. The padding is unnecessary and potentially costly; it does not establish that it caused the 429 responses. |
| The production closed basis permits simple low-degree terms and limited unary functions | Exact recovery of a target using ratios, logarithms, square roots or nested transforms can lie outside the registered hypothesis space. |
| Metadata has unbound scientific parameters and some malformed expressions | Literal exact recovery is unresolved for unbound parameters; malformed expressions must be marked not evaluable rather than automatically counted as a failure. A topology match does not establish fitted parameter equality. |

The previous predictive log-score PASS and this structural recovery NO-GO ask different questions. Neither result changes the other.

## Implemented repairs

* A separate, opt-in finite formula compiler accepts a bounded AST with exp, log, sqrt, Abs, division, fractional powers and compositions. It freezes constants inside transforms and refits only outer linear amplitudes. Its tokens are checked again on use, and domain/overflow errors fail closed. The old closed-basis model remains the default.
* `freeze_expanded_formula_model` explicitly names the new coefficient contract and creates a distinct frozen identity. Synthetic batch/sequential posterior checks exercise that path. It is a reference implementation, not an unknown-continuous-parameter search or a production measured decision executor.
* The opt-in Scientist review can now return a lineage-bound `COMPOSE_FORMULA_AST` directive. The prompt supplies indexed witnessed terms; code compiles only these terms and bounded operator nodes. Invalid expanded directives cannot silently become code-owned legacy fallback directives. The inner proposer, refit, source admission, support deduplication and frozen posterior use the same `expanded-formula-ast-v1` candidate language. Candidate numerical amplitudes do not define posterior inner parameters.
* The versioned recovery evaluator is in mainline. It separates structural topology, literal exact equality, and exact equality after independently registered physical parameter values have been substituted. Unbound values remain unresolved. Its candidate side is checked against the finite-bank grammar; truth may contain unbound parameters.
* A prospective benchmark package removes prompt padding, records provider failure separately, stops on any non-2xx generation ledger, uses the new admission and bank projection, and contains a frozen task-level bank-contrast gate before truth opening. Original scripts and config remain byte-identical in the package.
* The prospective driver still intentionally blocks execution after static preflight: the uploaded experimental files omit dependencies from the user's full local benchmark branch, so the adapter cannot be run or source-replay verified here. Editing the freeze JSON cannot bypass that source-level block. This remains a correctness implementation, not a validated experiment.

## Conditions for a new development run

1. Restore the full experimental source branch and verify the actual `DRRBenchmarkSearcher` config forwarding and artifact schema against the opt-in pipeline. Inner numeric parameters remain frozen in the candidate identity; a separately specified probabilistic model would be required to infer them as continuous unknowns.
2. Check provider account quota, concurrency and 429 response bodies outside the experiment. Run a response-free transport preflight, fix the request rate/payload using a new prospective version and freeze the handling of failed requests. Keep zero-request scientific abstentions separate from attempted-but-failed transport.
3. Register a variable mapping for each task and applicability rules for unbound or malformed truth expressions before opening new truth. Do not rename structural topology as parameter-instantiated exact recovery.
4. Freeze candidate budget, independent admission split, paired bank-contrast minimum, evaluable denominator, tasks/seeds and failure handling. Require enough *independent tasks* with different banks in each comparison arm; repeated seeds alone are not independent evidence.
5. Run new development tasks only after the correctness gates, then freeze a method and run untouched confirmation once. Do not turn these 36 opened rows into a second confirmation set or retroactively change their PASS/FAIL decision.

The attached benchmark scripts come from a local source branch not present in the public benchmark snapshot. The published snapshot lacks `drr_adapter.split_three_arm_training_samples` and the local three-arm protocol runner, so the prospective benchmark driver cannot be executed from only the historical scripts and public branch. The generator now verifies its input symbol mapping against the frozen evaluator mapping before proposing formulae; this boundary has not been replayed against the missing local branch.
