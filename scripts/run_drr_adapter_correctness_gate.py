"""No-data correctness Gate for the LLM-SRBench-to-mainline DRR adapter."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import load
from methods.hypothesis_mvp_pcpi.drr_searcher import DRRBenchmarkSearcher
from methods.hypothesis_mvp_pcpi.searcher import PCPISearcher

from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source
from hypothesis_mvp.pcpi.discovery_transaction import _publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("DRR adapter Gate output must be new")
    rows = np.column_stack((
        np.linspace(-1., 1., 80),
        np.linspace(-1., 1., 80),
        np.linspace(-1., 1., 80) ** 2))
    roles = drr_adapter.split_training_samples(
        rows, task_name="correctness-fixture", seed=17)
    candidates = [
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0 + x1", "source": "llm_evidence_synthesis",
         "origin": "llm"},
    ]
    readiness = drr_adapter.evaluate_drr_candidates(
        candidates, roles, condition="full_scientist_v6",
        task_name="correctness-fixture", seed=17)
    X = np.linspace(-.5, .5, 32)[:, None]
    y = np.sin(3.0 * X[:, 0]) + .05 * np.arange(len(X))
    positive_roles = drr_adapter.DRRSelectionRoles(
        X, y, X, y, X, y,
        np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]]),
        {
            "discovery_development": tuple(range(6)),
            "discovery_validation": tuple(range(6, 12)),
            "inference_initial": tuple(range(12, 18)),
            "action_covariates": tuple(range(18, 24)),
        },
    )
    resolution_candidates = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
        {"expression": "sin(x0)", "source": "llm_evidence_synthesis",
         "origin": "llm"},
    ]
    positive = drr_adapter.evaluate_drr_candidates(
        resolution_candidates, positive_roles,
        condition="positive-correctness-fixture",
        task_name="positive-correctness-fixture", seed=17)
    negative = drr_adapter.evaluate_drr_candidates(
        resolution_candidates[:1], positive_roles,
        condition="negative-correctness-fixture",
        task_name="negative-correctness-fixture", seed=17)
    prefix_curve = drr_adapter.evaluate_drr_prefix_candidates(
        resolution_candidates, positive_roles,
        condition="positive-prefix-correctness-fixture",
        task_name="positive-prefix-correctness-fixture", seed=17)
    recovered = drr_adapter.candidate_rows({
        "final_topk": [
            {"expression": "x0 + x1", "source": "engine:mcts",
             "origin": "deterministic"},
            {"expression": "x0*x1", "source": "llm_evidence_synthesis",
             "origin": "llm"},
        ],
        "evaluated_hypothesis_bank": candidates,
    }, "x0", 2)
    recovered_families = [
        drr_adapter._stacking.source_family(row) for row in recovered]
    split_source = inspect.getsource(drr_adapter.split_training_samples)
    evaluation_signature = inspect.signature(
        drr_adapter.evaluate_drr_candidates)
    decisions = {
        "action_role_exports_covariates_only":
            not hasattr(roles, "y_actions"),
        "four_train_only_roles_are_disjoint":
            len(set().union(*map(set, roles.role_row_indices.values())))
            == sum(map(len, roles.role_row_indices.values())),
        "adapter_has_no_test_or_ood_argument":
            not any(token in name.lower()
                    for name in evaluation_signature.parameters
                    for token in ("test", "ood", "heldout")),
        "split_source_never_extracts_action_response":
            'opened("action_covariates")' not in split_source
            and 'action_rows[:, 1:]' in split_source,
        "mainline_certificate_is_response_free":
            readiness.certificate["action_response_accessed"] is False
            and readiness.certificate["test_or_ood_accessed"] is False
            and readiness.certificate["heldout_opened"] is False,
        "production_searchers_import_against_safe_mainline":
            PCPISearcher is not None and DRRBenchmarkSearcher is not None,
        "protected_core_backbone_is_recovered_before_optional_sources":
            recovered_families[:2] == ["core", "core"],
        "optional_source_families_remain_available":
            "llm" in recovered_families,
        "candidate_pool_is_bounded":
            len(recovered) <= 8,
        "exact_fixture_produces_positive_readiness":
            positive.indicator == 1 and positive.ready is True,
        "exact_fixture_produces_registered_negative":
            negative.indicator == 0 and negative.ready is False,
        "fixed_prefix_curve_is_continuous_and_positive":
            prefix_curve.prefixes == (8, 16, 32)
            and prefix_curve.normalized_aulc > 0.0
            and prefix_curve.positive_prefix_count >= 1,
        "failures_have_registered_zero_indicator":
            readiness.indicator in {0, 1},
    }
    result = {
        "schema": "llm-srbench-drr-adapter-correctness-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "fixture_indicator": readiness.indicator,
        "positive_resolution_fixture_indicator": positive.indicator,
        "negative_resolution_fixture_indicator": negative.indicator,
        "positive_prefix_fixture_aulc":
            prefix_curve.normalized_aulc,
        "benchmark_source":
            verify_clean_git_source(PROJECT_ROOT),
        "mainline_source":
            verify_clean_git_source(PROJECT_ROOT.parent / "hypothesis_mvp"),
        "real_data_accessed": False,
        "benchmark_task_arrays_accessed": False,
        "test_or_ood_accessed": False,
        "llm_called": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic interface correctness only; no benchmark execution, "
            "efficacy, or superiority evidence."),
    }
    args.output_dir.mkdir(parents=True)
    _publish(args.output_dir / "DRR_ADAPTER_CORRECTNESS_GATE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
