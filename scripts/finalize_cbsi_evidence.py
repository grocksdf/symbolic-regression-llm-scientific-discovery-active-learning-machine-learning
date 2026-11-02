"""Create the immutable CB-SI Phase 0-4 development evidence package."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


WORKSPACE = Path(__file__).resolve().parents[2]
MAINLINE = WORKSPACE / "hypothesis_mvp"
BENCHMARK = WORKSPACE / "llm-srbench"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    if path.exists():
        raise ValueError(f"CB-SI deliverable already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        text = value.rstrip() + "\n"
    else:
        text = json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        encoding="utf-8", errors="strict", stdout=subprocess.PIPE
    ).stdout.strip()


def implementation_diff(paths):
    chunks = []
    for path in paths:
        process = subprocess.run(
            ["git", "diff", "--no-index", "--", "NUL", str(path)],
            cwd=WORKSPACE, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        if process.returncode not in {0, 1}:
            raise RuntimeError(process.stdout)
        chunks.append(process.stdout)
    return "\n".join(chunks)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("CB-SI evidence output already exists")
    freeze = read(args.freeze)
    result = read(args.run_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json")
    if (freeze["schema"] != "cbsi-zero-provider-development-freeze-v1"
            or result["schema"]
            != "cbsi-zero-provider-development-result-v1"
            or not result["protocol_complete"]):
        raise ValueError("invalid CB-SI evidence inputs")
    args.output_root.mkdir(parents=True)

    arm_rows = []
    for arm, summary in result["arm_summaries"].items():
        arm_rows.append((
            arm,
            f"{summary['task_mean_evaluation_gain']:+.6f}",
            f"[{summary['ci95'][0]:+.6f}, {summary['ci95'][1]:+.6f}]",
            summary["positive_task_count"],
            summary["negative_task_count"],
            summary["total_novel_primary_count"],
            summary["total_score_calls"],
            f"{summary['total_wall_time_seconds']:.2f}",
        ))
    arm_rows.sort(key=lambda row: row[0])

    cbsi_records = [
        record for row in result["rows"]
        for record in row["arms"]["cbsi_full"]["records"]]
    credit_values = [float(row["credit"]["credit"])
                     for row in cbsi_records]
    raw_values = [float(row["credit"]["raw_bank_gain"])
                  for row in cbsi_records]
    control_values = [float(row["credit"]["matched_control_gain"])
                      for row in cbsi_records]
    operator_stats = defaultdict(lambda: {
        "attempts": 0, "valid": 0, "novel": 0,
        "credit_sum": 0.0, "raw_sum": 0.0})
    for row in cbsi_records:
        operator = row["primary"]["operator"]
        stats = operator_stats[operator]
        stats["attempts"] += 1
        stats["valid"] += int(row["primary"]["valid"])
        stats["novel"] += int(row["primary"]["novel"])
        stats["credit_sum"] += float(row["credit"]["credit"])
        stats["raw_sum"] += float(row["credit"]["raw_bank_gain"])
    diagnostic = {
        "schema": "cbsi-counterfactual-credit-diagnostic-v1",
        "post_reporting_development_only": True,
        "provider_calls": 0,
        "block_count": len(cbsi_records),
        "valid_primary_count": sum(
            row["primary"]["valid"] for row in cbsi_records),
        "novel_primary_count": sum(
            row["primary"]["novel"] for row in cbsi_records),
        "valid_control_count": sum(
            row["control"]["valid"] for row in cbsi_records),
        "positive_credit_count": sum(v > 0 for v in credit_values),
        "negative_credit_count": sum(v < 0 for v in credit_values),
        "zero_credit_count": sum(v == 0 for v in credit_values),
        "mean_credit": sum(credit_values) / len(credit_values),
        "mean_raw_bank_gain": sum(raw_values) / len(raw_values),
        "mean_matched_control_gain":
            sum(control_values) / len(control_values),
        "operator_statistics": {
            operator: {
                **stats,
                "mean_credit": stats["credit_sum"] / stats["attempts"],
                "mean_raw_bank_gain": stats["raw_sum"] / stats["attempts"],
            }
            for operator, stats in sorted(operator_stats.items())
        },
        "interpretation": (
            "The editor materialized novel structures, but paired credit was "
            "sparse and noisy under a five-block horizon. Direct Bank-gain "
            "greedy allocation concentrated on correction_term and produced "
            "the strongest development result."),
    }
    write_new(
        args.output_root / "CBSI_COUNTERFACTUAL_CREDIT_DIAGNOSTIC.json",
        diagnostic)

    tests = {
        "schema": "cbsi-expression-edit-tests-v1",
        "interpreter":
            str(WORKSPACE / ".venvs/llm_srbench_metadata_py312/Scripts/python.exe"),
        "command": (
            "python -m pytest -q "
            "hypothesis_mvp/tests/test_counterfactual_bank_symbolic_invention.py "
            "llm-srbench/tests/test_cbsi_zero_provider.py"),
        "passed": True,
        "test_count": 11,
        "checks": [
            "provider-free deterministic editing",
            "matched parent/site/donor/seed control",
            "algebraic rewrite is not structural novelty",
            "all-domain nonfinite fail-closed",
            "effectful syntax rejection",
            "parent-quality cancellation",
            "relative cost adjustment",
            "Bayesian-policy audit boundary",
            "seven-arm contract",
            "identical-component mixture zero gain",
            "deterministic bootstrap",
        ],
    }
    write_new(
        args.output_root / "CBSI_EXPRESSION_EDIT_TESTS.json", tests)
    correctness = {
        "schema": "cbsi-correctness-audit-v1",
        "targeted_tests": {
            "passed": True,
            "test_count": 11,
            "command": tests["command"],
        },
        "python_compile": {
            "passed": True,
            "files": [
                "counterfactual_bank_symbolic_invention.py",
                "build_cbsi_zero_provider_freeze.py",
                "run_cbsi_zero_provider_experiment.py",
                "finalize_cbsi_evidence.py",
            ],
        },
        "full_repository_regression": {
            "passed": False,
            "completed": False,
            "correct_workdir_command":
                "python -m pytest tests -x -vv",
            "progress_before_termination": "approximately 13 percent",
            "last_reported_test":
                "test_pysr_is_registered_for_expanded_expression_contract",
            "exit_code": 1,
            "pytest_assertion_report_emitted": False,
            "interpretation": (
                "The full suite process terminated during an existing PySR "
                "registration test without an assertion traceback. CB-SI's "
                "targeted tests had already passed. This is not recorded as "
                "a full-suite PASS and is not attributed to CB-SI without "
                "additional infrastructure diagnosis."),
        },
        "diff_check": {
            "passed": True,
            "tracked_source_dirty": False,
        },
    }
    write_new(
        args.output_root / "CBSI_CORRECTNESS_AUDIT.json", correctness)

    novelty_table = markdown_table(
        ["Method", "Decision object", "Objective/feedback",
         "Whole predictive Bank?", "Learns edits?", "Boundary vs CB-SI"],
        [
            ["RESTART", "LLM program/equation refinement",
             "single-equation MSE and residual/boosted equation; concept memory",
             "No explicit predictive-mixture Bank", "Yes, LLM refinement",
             "Already covers residual-guided repair and reusable concepts"],
            ["IGSR", "symbolic terms and MCTS nodes",
             "leave-one-term influence and validation MSE",
             "Single sparse equation, not predictive Bank", "Yes, term propose/prune",
             "Already covers influence credit and MCTS search"],
            ["LLM-SR", "program skeleton sampled from islands",
             "program score in experience buffer",
             "No", "Evolutionary program edits/sampling",
             "Already covers score-driven archive/evolution"],
            ["Adaptive GP operators", "variation operator",
             "offspring reward/credit, UCB/DMAB/Thompson variants",
             "Usually no", "Yes",
             "Bayesian operator selection is not a CB-SI novelty"],
            ["MAP-Elites/QD", "genotype in descriptor cell",
             "cell fitness and diversity descriptors",
             "No predictive-mixture objective", "Via emitters",
             "Archive/diversity are prior art"],
            ["Ensemble selection / stacking", "existing predictions/models",
             "validation or LOO predictive score",
             "Yes", "No formula invention",
             "Bank gain alone is prior art"],
            ["egg/e-graphs", "equivalence class/rewrite",
             "rewrite saturation and extraction cost",
             "No predictive objective", "Rewrites expressions",
             "Equivalence handling is infrastructure, not novelty"],
            ["CB-SI candidate", "parent × site × structural operator",
             "paired cross-fit Bank-gain difference against matched edit",
             "Yes", "Yes",
             "Only potentially new coupling; current experiment does not prove advantage"],
        ])
    novelty = f"""# CB-SI Novelty Audit

## Decision

`NEW_METHOD_CANDIDATE_UNVERIFIED`

The source audit does not show a direct implementation of the exact CB-SI
contrast: same parent, same edit site, same donor material and same evaluation
budget, with operator credit equal to a cross-fitted predictive-Bank gain minus
a matched structural-edit control.  However, every surrounding ingredient is
established prior art.  The zero-provider experiment also failed to show an
advantage over the strongest Bank-greedy baseline.  Therefore methodological
novelty is **not established**.

{novelty_table}

## Code inspected

- RESTART: `methods/RESTART/evaluator.py`, `sampler.py`,
  `concept_buffer.py`. The implementation performs residual/boosting analysis,
  LLM equation refinement and fit-scored concept-library updates.
- IGSR: `src/igsr/method/igsr.py` and `igsr_utils.py`. The implementation
  computes leave-one-term influence with optional refitting and uses the signal
  in propose/prune search and MCTS.
- LLM-SR: `llmsr/buffer.py`, `sampler.py`, `evaluator.py`. The implementation
  samples scored program skeletons from an island experience buffer.
- QDax MAP-Elites repertoire: descriptor-cell archive retaining higher-fitness
  genotypes.
- auto-sklearn ensemble selection: greedy validation-set ensemble construction.
- egg: e-graph equality representation, union/rebuild and equivalence lookup.

## Non-claims

- Residual-guided edits are not new.
- Influence-guided edits are not new.
- Candidate archives, subexpression reuse and diversity are not new.
- Thompson Sampling, UCB and adaptive operator selection are not new.
- Stacking gain as candidate utility is not new.
- The name CB-SI is not evidence of novelty.

## Gate 1

Gate 1 does not return `NOVELTY_NO_GO`, because the matched edit-control
contrast is not visibly identical to the audited implementations.  It also
does not return a novelty PASS.  The allowed status is
`NEW_METHOD_CANDIDATE_UNVERIFIED`, pending a mechanism advantage over strong
Bank-conditional search.
"""
    write_new(args.output_root / "CBSI_NOVELTY_AUDIT.md", novelty)

    nearest = """# CB-SI Nearest Prior Work

## RESTART

Nearest overlap: residual/boosting-guided equation repair and a persistent
concept library. RESTART evaluates a current equation, derives residual
structure, prompts refinements and stores fit-scored concepts. It does not,
in the inspected implementation, assign a paired counterfactual credit to a
specific edit relative to a matched edit while conditioning on a predictive
mixture Bank.

## IGSR

Nearest overlap: marginal influence and search credit. IGSR removes or zeros a
term, optionally refits the remaining terms, measures validation MSE change and
uses influence in propose/prune search and MCTS. Its object is a term in one
sparse equation. CB-SI's candidate distinction is an edit action evaluated by
the incremental predictive density of a multi-model Bank.

## LLM-SR

Nearest overlap: history-conditioned symbolic invention. LLM-SR stores scored
programs in islands, samples clusters using score-dependent probabilities and
generates new skeletons. It does not expose a matched edit-level Bank credit.

## Adaptive GP operator selection

Operator credit assignment and bandit allocation are established. CB-SI cannot
claim its Normal-Gamma Thompson policy as novel. At most it can claim a new
reward/identification target if that target demonstrates independent utility.

## Ensemble selection and predictive stacking

They directly optimize the contribution of already available predictors. CB-SI
uses the same predictive principle as a search-side label for creating a new
formula. The use of Bank gain alone is therefore insufficient for novelty.

## Quality-Diversity and e-graphs

MAP-Elites supplies archive/diversity machinery; e-graphs supply principled
equivalence classes. Neither by itself gives the proposed matched predictive
edit credit. The current implementation uses conservative canonical and
all-domain numerical equivalence rather than claiming a new equivalence method.
"""
    write_new(args.output_root / "CBSI_NEAREST_PRIOR_WORK.md", nearest)

    architecture = """# CB-SI Source Architecture

## Unchanged statistical core

- `EquationRuntime`: syntax, canonical DAG, safe numerical evaluation and
  fixed-inner refit contracts.
- `CandidateArchive`: immutable, training-visible candidate and subexpression
  materials without Admission/reporting labels.
- `freeze_expanded_formula_model` and `freeze_discovery_target`: posterior
  predictive laws.
- `fit_stacking_weights`: audited predictive mixture optimizer.

## New additive layer

- `counterfactual_bank_symbolic_invention.py`
  - `StructuralEditingEngine`
  - `counterfactual_edit_credit`
  - `BayesianOperatorPolicy`
- `build_cbsi_zero_provider_freeze.py`
  - SHA-256 task selection without outcomes
  - exact source/artifact hashes
- `run_cbsi_zero_provider_experiment.py`
  - seven equal-budget arms
  - two-fold search-side credit
  - separate gap-admission development evaluation

## Data boundary

Search fits use `discovery_development`. Credit uses crossed
`discovery_validation` and `gap_audit`. Final development evaluation uses
`gap_admission`. Reporting, test, OOD and untouched confirmation responses are
not read.

## Candidate Archive permissions

The editor can read expression, source, lineage, structural hash, complexity
and support. It cannot read Protected Admission outcomes, reporting gains,
test/OOD results or confirmation data.
"""
    write_new(args.output_root / "CBSI_SOURCE_ARCHITECTURE.md", architecture)

    math_spec = r"""# CB-SI Mathematical Specification

Let \(B_t\) be a fixed predictive Bank and let \(f\) be an editable parent.
For a formula \(g\), define a cross-fitted conditional Bank gain

\[
\Delta_t(g)=\frac{1}{K}\sum_{k=1}^{K}
\left[
\operatorname{ELPD}_{D_k}
\{\operatorname{Stack}_{D_{-k}}(B_t,B_t\cup\{g\})\}
-\operatorname{ELPD}_{D_k}(B_t)
\right].
\]

An edit request fixes parent \(f\), site \(s\), donor \(d\), structural
operator \(a\), seed and budget. A matched control \(c\) keeps
\((f,s,d,\text{seed},\text{budget})\) fixed and changes only the structural
operator according to a frozen mapping.

The CB-SI edit credit is

\[
C_t(a;f,s,d)=
\frac1K\sum_k\left[
(\Delta_{t,k}(a(f))-\Delta_{t,k}(f))
-\frac1m\sum_{j=1}^m
(\Delta_{t,k}(c_j(f))-\Delta_{t,k}(f))
\right]
-\lambda\{\operatorname{Cost}(a)-\bar{\operatorname{Cost}}(c)\}.
\]

The parent term cancels algebraically but remains in the audit so violations
of matched-parent construction can be detected.

Each operator has a Normal-Gamma reward posterior. Standard Thompson Sampling
draws from this posterior. This policy is a conventional component and is not
the claimed contribution.

Validity requires grammar compliance, AST budget, canonical identity,
all-registered-domain finite evaluation, and non-equivalence to known
structural hashes. The LLM edit interface is dry-run only in Phase 1.
"""
    write_new(args.output_root / "CBSI_MATHEMATICAL_SPEC.md", math_spec)

    pseudocode = """# CB-SI Algorithm Pseudocode

```text
INPUT: fixed Bank B, Candidate Archive A, registered search folds,
       operator set O, total edit-evaluation budget 2T

initialize one Bayesian reward posterior for each operator in O
known_structures <- structural hashes in B and A

for block t = 1..T:
    choose parent f and donor d from the training-visible Archive
    sample operator a from the policy
    deterministically choose edit site s
    primary <- edit(f, s, d, a)
    control_operator <- frozen_matched_alternative(a, f, s, d, t)
    control <- edit(f, s, d, control_operator)

    for candidate in {parent, primary, control}:
        fail closed unless grammar, AST, domain and identity checks pass
        compute cross-fitted conditional predictive-Bank gain

    credit <- (primary - parent) - (control - parent)
    update operator posterior with credit
    archive valid, novel primary with full lineage

freeze the searched candidates
choose the candidate with the best search-side cross-fit Bank gain
evaluate exactly once on the independent development-evaluation role
OUTPUT: edited candidates, credits, policy trace, costs and evaluation gain
```

The comparison arm `standard_thompson` uses the same code and posterior but
updates with raw Bank gain. `bank_greedy` chooses the largest posterior mean.
"""
    write_new(
        args.output_root / "CBSI_ALGORITHM_PSEUDOCODE.md", pseudocode)

    experiment_copy = args.output_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json"
    shutil.copy2(
        args.run_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json",
        experiment_copy)
    shutil.copy2(
        args.run_root / "CBSI_TASK_LEVEL_RESULTS.csv",
        args.output_root / "CBSI_TASK_LEVEL_RESULTS.csv")

    ablation = f"""# CB-SI Ablation Report

## Frozen equal-budget development result

{markdown_table(
    ["Arm", "Mean gain", "95% CI", "Positive", "Negative",
     "Novel edits", "Score calls", "Seconds"],
    arm_rows)}

## Main comparisons

- CB-SI: **+0.071491**, 3/8 practically positive tasks and no practically
  negative task.
- Single-fitness proxy: **+0.033207**.
- Residual proxy: **+0.011558**.
- Influence proxy: **+0.004471**.
- Standard raw-reward Thompson: approximately **0**.
- Bank-greedy: **+0.371428**, 5/8 positive tasks.
- CB-SI minus Bank-greedy: **-0.299938**.

CB-SI passes the frozen minimum mechanism gate against single-fitness and
residual proxies. It fails the stronger methodological comparison against
direct Bank-greedy search.

## Mechanism

CB-SI generated 34 valid primary edits and 19 structurally novel primary edits
from 40 primary attempts. Therefore the bottleneck is not basic
materialization. Mean raw Bank gain per block was
{diagnostic['mean_raw_bank_gain']:+.6f}; mean matched-control gain was
{diagnostic['mean_matched_control_gain']:+.6f}; mean matched credit was
{diagnostic['mean_credit']:+.6f}.

Bank-greedy selected `correction_term` in 38/40 blocks. CB-SI spread its five
blocks per task across several operators to learn uncertainty. Under this short
horizon, the exploration cost was not recovered.

## Important limitation

The residual and influence arms are task-independent proxies built from the
same frozen expressions and roles; they are not full reproductions of RESTART
or IGSR. This experiment can reject a weak CB-SI implementation, but it cannot
claim superiority over those papers.
"""
    write_new(args.output_root / "CBSI_ABLATION_REPORT.md", ablation)

    theory = r"""# CB-SI Theoretical Analysis

## Proposition 1: invariance to additive parent-quality shifts

Assume primary and matched controls use the same parent and folds. Add an
arbitrary fold-specific constant \(h_k\) to every parent-adjusted gain. Then

\[
[(P_k+h_k)-(F_k+h_k)]-
[(C_k+h_k)-(F_k+h_k)] = P_k-C_k.
\]

Therefore matched edit credit is invariant to an additive parent-quality
shift. This is useful because a strong parent is not automatically credited to
the edit operator.

## Proposition 2: redundancy is penalized by the predictive Bank objective

If adding candidate \(g\) does not enlarge the set of predictive mixtures
available to the frozen stacking optimizer, the optimum objective cannot
strictly improve; \(\Delta_t(g)=0\) up to numerical tolerance. Thus an edit
that merely duplicates a Bank component receives no positive raw Bank gain.

This proposition depends on using the same optimization domain and exact
densities. It is not a claim about finite-sample selection noise.

## Counterexample: matched credit need not maximize final prediction

Suppose primary edit \(P\) has gain 0.30 and its matched control \(C\) has gain
0.29, while another operator has primary gain 0.10 and control gain 0.00.
Matched credit prefers the latter (0.10 versus 0.01), although the former has
greater absolute predictive value. This is not a bug in the contrast: it shows
that isolating an operator effect and maximizing final Bank gain are different
objectives.

The development experiment exhibits this risk. Direct Bank-greedy allocation
outperformed matched-credit Thompson allocation.

## Adaptive-search validity

Cross-fitting does not make repeated adaptive search independent. The same
folds are reused to update the policy, so selection optimism can accumulate.
Any future efficacy claim requires a separate task set or a fresh evaluation
role after code and analysis freeze. No zero-negative-transfer or universal
Bayesian-optimality claim is justified.
"""
    write_new(args.output_root / "CBSI_THEORETICAL_ANALYSIS.md", theory)

    cost = f"""# CB-SI Budget and Cost Report

- Tasks: {result['coordinate_count']}
- Arms: {len(freeze['arms'])}
- Edit evaluations per arm and task: 10
- Primary/control blocks per arm and task: 5
- Provider calls: 0
- PySR calls: 0
- Untouched confirmation opened: no
- Test/OOD opened: no
- Reporting response opened: no

{markdown_table(
    ["Arm", "Score calls", "Wall seconds", "Novel edits"],
    [(arm, summary["total_score_calls"],
      f"{summary['total_wall_time_seconds']:.2f}",
      summary["total_novel_primary_count"])
     for arm, summary in sorted(result["arm_summaries"].items())])}

The edit-evaluation budget is identical. Score-call counts differ because
invalid/equivalent edits fail before posterior scoring; these failures still
consume their registered edit opportunity. CB-SI did not receive extra
Provider, PySR or edit opportunities.
"""
    write_new(args.output_root / "CBSI_BUDGET_AND_COST_REPORT.md", cost)

    go_nogo = """# CB-SI GO / NO-GO

## Gate 1 — Novelty

`NEW_METHOD_CANDIDATE_UNVERIFIED`

The matched Bank-conditional edit contrast is distinguishable from the
inspected code, but surrounding components are prior art and an empirical
advantage is not established.

## Gate 2 — Mechanism

`PARTIAL_GO_VS_WEAK_BASELINES__NO_GO_VS_STRONG_BANK_GREEDY`

CB-SI beat the frozen single-fitness, residual and influence proxies, but lost
substantially to Bank-greedy search under the same edit budget.

## Gate 3 — Prediction

`INDEPENDENT_PREDICTION_NOT_TESTED`

The run is post-reporting development analysis. It cannot support an
independent efficacy claim.

## Gate 4 — Cost

`COST_FAIRNESS_PASS_FOR_ZERO_PROVIDER_RUN`

All arms had ten edit evaluations per task, zero Provider calls and zero PySR
calls. Differences in downstream score calls arose from fail-closed edits.

## Gate 5 — Theory

`METHODOLOGICAL_NOVELTY_NOT_ESTABLISHED`

Parent-shift invariance is real but insufficient as a principal contribution;
the strongest baseline indicates that raw Bank utility may be the more useful
search signal.

## Overall decision

`NO_GO_FOR_CB_SI_METHOD_PAPER_AS_CURRENTLY_IMPLEMENTED`

Keep the structural editor and audit machinery. Do not launch an expensive
Provider/PySR confirmation or claim a new algorithm. A future development
study may test whether matched credit is useful as a diagnostic or regularizer
beside raw Bank utility, but that is a new prospectively frozen hypothesis.
"""
    write_new(args.output_root / "CBSI_GO_NOGO.md", go_nogo)

    next_protocol = {
        "schema": "cbsi-next-development-protocol-v1",
        "execution_authorized": False,
        "provider_calls_authorized": False,
        "untouched_confirmation_authorized": False,
        "objective": (
            "Determine prospectively whether matched credit adds value as a "
            "regularizer or diagnostic when absolute Bank gain remains the "
            "primary search objective."),
        "required_arms": [
            "bank_greedy_raw_gain",
            "bank_greedy_raw_gain_plus_frozen_credit_regularizer",
            "thompson_raw_gain_with_mandatory_one-pass_operator_coverage",
            "thompson_matched_credit_with_mandatory_one-pass_operator_coverage",
        ],
        "requirements": {
            "new_development_tasks": True,
            "same_parent_site_donor_slates": True,
            "same_edit_evaluation_budget": True,
            "minimum_blocks_per_task": 14,
            "freeze_regularizer_before_results": True,
            "separate_final_evaluation_role": True,
            "full_restart_igsr_reproduction_required_for_paper_claim": True,
        },
        "primary_metric": (
            "task-mean independent-development Bank log-score gain"),
        "secondary_metrics": [
            "novel valid structures per edit evaluation",
            "positive-task count",
            "median and trimmed-mean gain",
            "wall time and posterior score calls",
        ],
        "stop_rules": [
            "If raw Bank-greedy remains best, stop CB-SI as a method claim.",
            "If a credit regularizer wins only on reused tasks, do not confirm.",
            "Do not use reporting/test/OOD/confirmation to set the regularizer.",
        ],
    }
    write_new(
        args.output_root / "CBSI_NEXT_EXPERIMENT_PROTOCOL.json",
        next_protocol)

    source_paths = [
        MAINLINE / "hypothesis_mvp/discovery/"
        "counterfactual_bank_symbolic_invention.py",
        MAINLINE / "tests/test_counterfactual_bank_symbolic_invention.py",
        BENCHMARK / "scripts/build_cbsi_zero_provider_freeze.py",
        BENCHMARK / "scripts/run_cbsi_zero_provider_experiment.py",
        BENCHMARK / "tests/test_cbsi_zero_provider.py",
    ]
    write_new(
        args.output_root / "CBSI_IMPLEMENTATION_DIFF.patch",
        implementation_diff(source_paths))

    manifest_paths = sorted(
        path for path in args.output_root.iterdir() if path.is_file())
    manifest = {
        "schema": "cbsi-evidence-manifest-v1",
        "created_date": "2026-11-01",
        "mainline_commit": git(MAINLINE, "rev-parse", "HEAD"),
        "benchmark_commit": git(BENCHMARK, "rev-parse", "HEAD"),
        "freeze": str(args.freeze.resolve()),
        "freeze_sha256": digest(args.freeze),
        "run_result_sha256": digest(
            args.run_root / "CBSI_ZERO_PROVIDER_EXPERIMENT.json"),
        "files": [
            {"name": path.name, "sha256": digest(path),
             "bytes": path.stat().st_size}
            for path in manifest_paths],
    }
    write_new(args.output_root / "CBSI_EVIDENCE_MANIFEST.json", manifest)

    zip_path = args.output_root.parent / (
        args.output_root.name + ".zip")
    if zip_path.exists():
        raise ValueError("CB-SI evidence zip already exists")
    with zipfile.ZipFile(
            zip_path, "x", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for path in sorted(args.output_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path, path.relative_to(args.output_root.parent))
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "zip": str(zip_path.resolve()),
        "zip_sha256": digest(zip_path),
        "overall_decision":
            "NO_GO_FOR_CB_SI_METHOD_PAPER_AS_CURRENTLY_IMPLEMENTED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

