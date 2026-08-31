"""Run the no-data P3J.3 refinement-equivalence and static-cost Gate."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import p3j_worst_case_cost_ledger
from hypothesis_mvp.pcpi import class_conditional_semiparametric as p3j
from hypothesis_mvp.pcpi import real_acquisition


CONFIG = PROJECT_ROOT / "configs" / "p3j_3_refinement_cost_gate.json"
RESULT_SCHEMA = "pcpi-p3j3-refinement-cost-gate-result-v1"


def _load(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema", "stage", "candidate_count_upper", "ambiguity_model_count",
        "maximum_class_count", "maximum_residual_leaf_count",
        "minimum_nodes_per_leaf", "maximum_nodes_per_leaf", "growth_factor",
        "refinement_policy", "cost_scope", "runner_composition_authorized",
        "operational_execution_authorized", "real_data_access",
        "candidate_response_access", "validation_response_access",
        "heldout_access", "simulated_experiment",
    }
    if set(config) != expected:
        raise ValueError("P3J.3 cost config fields changed")
    if (
        config["schema"] != "pcpi-p3j3-refinement-cost-gate-config-v1"
        or config["stage"] != "P3J.3"
        or config["refinement_policy"]
        != "preceding-fine-is-next-coarse-exact-reuse"
        or any(
            config[key]
            for key in (
                "runner_composition_authorized",
                "operational_execution_authorized",
                "real_data_access",
                "candidate_response_access",
                "validation_response_access",
                "heldout_access",
                "simulated_experiment",
            )
        )
    ):
        raise ValueError("P3J.3 authorization boundary changed")
    return config


def _evaluate(config: dict[str, object]) -> dict[str, object]:
    ledger = p3j_worst_case_cost_ledger(
        candidate_count=config["candidate_count_upper"],
        ambiguity_model_count=config["ambiguity_model_count"],
        maximum_class_count=config["maximum_class_count"],
        maximum_leaf_count=config["maximum_residual_leaf_count"],
        minimum_nodes_per_leaf=config["minimum_nodes_per_leaf"],
        maximum_nodes_per_leaf=config["maximum_nodes_per_leaf"],
        growth_factor=config["growth_factor"],
    )
    refinement_source = inspect.getsource(
        p3j.refine_class_conditional_semiparametric_eig
    )
    adaptive_source = inspect.getsource(
        real_acquisition._estimate_class_conditional_maximin_until_ranked
    )
    decisions = {
        "preceding_fine_grid_is_bound_as_next_coarse": all(
            token in refinement_source
            for token in (
                "order != 2 * preceding.nodes_per_leaf",
                "scores - preceding.scores",
                "residual_state_hash",
                "target_partition_hash",
            )
        ),
        "adaptive_loop_uses_refinement_after_first_look": all(
            token in adaptive_source
            for token in (
                "preceding = estimates",
                "refine_class_conditional_semiparametric_eig",
            )
        ),
        "static_schedule_is_complete": ledger.orders == (32, 64, 128, 256, 512),
        "reuse_strictly_reduces_node_work": (
            ledger.reused_source_class_nodes < ledger.naive_source_class_nodes
        ),
        "worst_case_cross_class_work_is_exposed": (
            ledger.reused_cross_class_density_evaluations == 101_154_816
        ),
        "runner_remains_blocked_pending_chunked_checkpointed_composition": (
            not config["runner_composition_authorized"]
            and not config["operational_execution_authorized"]
        ),
    }
    if not all(decisions.values()):
        raise AssertionError("P3J.3 refinement/cost Gate failed")
    return {
        "schema": RESULT_SCHEMA,
        "stage": "P3J.3",
        "status": "passed-refinement-cost-audit-runner-composition-blocked",
        "decisions": decisions,
        "orders": list(ledger.orders),
        "naive_source_class_nodes": ledger.naive_source_class_nodes,
        "reused_source_class_nodes": ledger.reused_source_class_nodes,
        "saved_source_class_nodes": ledger.saved_source_class_nodes,
        "savings_fraction": ledger.savings_fraction,
        "reused_cross_class_density_evaluations": (
            ledger.reused_cross_class_density_evaluations
        ),
        "runner_composition_authorized": False,
        "operational_execution_authorized": False,
        "real_data_access": False,
        "candidate_response_access": False,
        "validation_response_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "formal_efficacy_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(_evaluate(_load(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
