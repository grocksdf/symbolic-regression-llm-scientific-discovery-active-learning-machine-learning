"""End-to-end synthetic Scientist materialization Gate for four formula families."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "hypothesis_mvp"))
sys.path.insert(0, str(ROOT / "scripts"))

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.agent import (
    DiscoveryAgent, DiscoveryAgentConfig,
)
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model,
)
from hypothesis_mvp.discovery.proposal_runtime import (
    ProposalRuntime, ProviderRoute, ProviderSettings,
)
from hypothesis_mvp.discovery.scientist_policy import deterministic_plan
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from expanded_formula_admission_v2 import (
    project_expanded_admission_arms,
)
from formula_bank_materialization_v2 import expanded_bank_rows
from formula_recovery_contract import assess_formula_recovery


def _ref(lineage, index=0):
    return {"ref": {"lineage_id": lineage, "term_index": index}}


CASES = {
    "x_exp_abs": {
        "n_features": 1,
        "truth": "x0*exp(-Abs(x0))",
        "evidence": [
            {"engine": "polynomial_lasso", "lineage_id": "a",
             "expression": "x0"},
            {"engine": "sparse_library", "lineage_id": "b",
             "expression": "x0"},
        ],
        "tree": {"op": "mul", "args": [
            _ref("a"), {"op": "exp", "args": [
                {"op": "neg", "args": [
                    {"op": "Abs", "args": [_ref("b")]}]}]}]},
    },
    "log_one_plus_square": {
        "n_features": 1,
        "truth": "log(1+x0**2)",
        "evidence": [
            {"engine": "polynomial_lasso", "lineage_id": "a",
             "expression": "x0"},
            {"engine": "sparse_library", "lineage_id": "b",
             "expression": "x0"},
        ],
        "tree": {"op": "log", "args": [
            {"op": "add", "args": [
                {"const": 1},
                {"op": "mul", "args": [_ref("a"), _ref("b")]}]}]},
    },
    "safe_ratio": {
        "n_features": 2,
        "truth": "x0/(1+x1**2)",
        "evidence": [
            {"engine": "polynomial_lasso", "lineage_id": "a",
             "expression": "x0"},
            {"engine": "sparse_library", "lineage_id": "b",
             "expression": "x1"},
        ],
        "tree": {"op": "div", "args": [
            _ref("a"), {"op": "add", "args": [
                {"const": 1}, {"op": "pow", "args": [
                    _ref("b"), {"const": 2}]}]}]},
    },
    "scaled_sine": {
        "n_features": 1,
        "truth": "sin(1.7*x0)",
        "evidence": [
            {"engine": "polynomial_lasso", "lineage_id": "a",
             "expression": "x0"},
            {"engine": "sparse_library", "lineage_id": "b",
             "expression": "1"},
        ],
        "tree": {"op": "add", "args": [
            {"op": "sin", "args": [{"op": "mul", "args": [
                {"const": 1.7}, _ref("a")]}]},
            {"op": "mul", "args": [{"const": 0}, _ref("b")]}]},
    },
}


def _values(case, X):
    x0 = X[:, 0]
    if case == "x_exp_abs":
        return x0 * np.exp(-np.abs(x0))
    if case == "log_one_plus_square":
        return np.log1p(x0 ** 2)
    if case == "safe_ratio":
        return x0 / (1.0 + X[:, 1] ** 2)
    if case == "scaled_sine":
        return np.sin(1.7 * x0)
    raise ValueError("unknown synthetic formula case")


def _agent():
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso", "sparse_library"),
        engine_repeats=1, engine_budget=2, cycles=1,
        discovery_budget=16, search_iterations=2,
        discovery_islands=("novelty",),
        synthesis_evaluation_reserve=1, llm_evaluation_reserve=1,
        refit_policy="pcpi-expanded-fixed-inner-v1",
        scientist_orchestration=True, require_explicit_skill_controls=True,
        typed_evidence_synthesis=True, typed_inner_augmentation=True,
        expanded_formula_synthesis=True)
    settings = ProviderSettings(routes=(ProviderRoute(
        "https://fixture.invalid", "scripted-scientist", "fixture-key"),))
    return DiscoveryAgent(config, settings), settings


def _one(case_name, case):
    n_features = case["n_features"]
    agent, settings = _agent()
    planner = ProposalRuntime(
        EquationRuntime(
            n_features, refit_policy="pcpi-expanded-fixed-inner-v1"),
        n_features, settings, 1)
    raw_review = {
        "protocol_id": "scientific-engine-evidence-review-v1",
        "supported_mechanisms": ["scripted synthetic formula witness"],
        "contradicted_mechanisms": [], "cross_engine_conflicts": [],
        "synthesis_instructions": ["compile the registered typed formula AST"],
        "stop": False, "stop_reason": "continue",
        "synthesis_directives": [{
            "operation": "COMPOSE_FORMULA_AST",
            "lineage_ids": [row["lineage_id"] for row in case["evidence"]],
            "rationale": "registered end-to-end materialization fixture",
            "formula_ast": case["tree"],
        }],
    }
    calls = []
    def scripted_complete_json(*, system_message, payload):
        calls.append({"system_message_sha256": sha256(
            system_message.encode()).hexdigest(),
            "payload_sha256": sha256(json.dumps(
                payload, sort_keys=True, default=str).encode()).hexdigest()})
        return raw_review, {"scripted_provider": True}
    planner.complete_json = scripted_complete_json
    plan = deterministic_plan(
        ("polynomial_lasso", "sparse_library"), 2)
    review, telemetry = planner.review_engine_evidence(
        plan=plan, engine_evidence=case["evidence"],
        require_typed_synthesis=True, allow_interactions=True,
        expanded_formula_synthesis=True)
    candidates, synthesis = agent._compile_cycle_synthesis(
        review, case["evidence"], n_features)
    report = {
        "drr_candidate_rows": [
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic"},
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic"},
            *([] if n_features == 1 else [{
                "expression": "x1", "source": "engine:sparse_library",
                "origin": "deterministic"}]),
        ],
        "evaluated_hypothesis_bank": candidates,
    }
    core, optional, materialization = expanded_bank_rows(
        report, n_features, 4)
    grid = np.linspace(.2, 1.8, 64)
    X = (grid[:, None] if n_features == 1 else
         np.column_stack((grid, np.linspace(.3, 1.5, 64))))
    y = _values(case_name, X)
    fit = RoleDataset(DataRole.DEVELOPMENT, X[:24], y[:24])
    gap = RoleDataset(DataRole.VALIDATION, X[24:34], y[24:34])
    admission = RoleDataset(DataRole.VALIDATION, X[34:44], y[34:44])
    selector = RoleDataset(DataRole.VALIDATION, X[44:54], y[44:54])
    projection = project_expanded_admission_arms(
        core, optional, fit, gap, admission, selector, X[54:],
        n_features=n_features,
        identity=sha256(case_name.encode()).hexdigest())
    bank_rows = projection["arms"]["accept_all"]
    model = freeze_expanded_formula_model(
        bank_rows, n_features=n_features, prior=NormalInverseGammaPrior(),
        exploration_identity=sha256(case_name.encode()).hexdigest(),
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    posterior = model.engine(model.stable_hash).fit_batch(fit.X, fit.y)
    generated = candidates[0]["expression"] if candidates else ""
    recovery = assess_formula_recovery(
        generated, case["truth"],
        tuple(f"x{i}" for i in range(n_features)))
    checks = {
        "scientist_review_called": len(calls) == 1,
        "typed_directive_present":
            len(review.synthesis_directives) == 1,
        "ast_compiled": synthesis["compiled_candidate_count"] == 1,
        "materialized_novel_candidate":
            materialization["optional_materialized_count"] == 1,
        "admission_chain_completed":
            set(projection["arms"]) == {
                "independent_protected", "same_data", "accept_all"},
        "frozen_posterior_normalized":
            abs(posterior.probability_sum - 1.0) <= 2e-12,
        "literal_recovery": recovery["literal_exact"] is True,
        "structural_recovery":
            recovery["structural_topology"] is True,
    }
    return {
        "case": case_name, "truth": case["truth"],
        "generated_expression": generated,
        "checks": checks, "passed": all(checks.values()),
        "scientist_telemetry": telemetry,
        "synthesis_audit": synthesis,
        "materialization_audit": materialization,
        "admission_identity": projection["candidate_bank_identity"],
        "posterior_identity": model.stable_hash,
        "recovery": recovery,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("synthetic materialization Gate output must be new")
    rows = [_one(name, case) for name, case in CASES.items()]
    result = {
        "schema":
            "expanded-formula-v2-end-to-end-synthetic-materialization-gate-v1",
        "passed": all(row["passed"] for row in rows),
        "case_count": len(rows), "rows": rows,
        "actual_provider_called": False,
        "scripted_provider_used_for_deterministic_scientist_review": True,
        "scientist_orchestration_exercised": True,
        "compiler_only_unit_test": False,
        "benchmark_accessed": False, "scientific_data_accessed": False,
        "confirmation_accessed": False,
        "claim_boundary": (
            "Deterministic end-to-end Scientist materialization correctness "
            "only; provider transport is certified separately and no efficacy "
            "or benchmark claim is made."),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
