import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_cbsi_zero_provider_experiment.py")
SPEC = importlib.util.spec_from_file_location("cbsi_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_arm_modes_are_distinct_and_complete():
    assert set(MODULE.ARM_MODES) == {
        "blind_random",
        "single_fitness_proxy",
        "residual_proxy",
        "influence_proxy",
        "bank_greedy",
        "standard_thompson",
        "cbsi_full",
        "cbsi_feasible",
        "cbsi_frontier",
    }
    assert MODULE.ARM_MODES["cbsi_full"] == "thompson_matched"


def test_mixture_gain_is_zero_for_identical_component():
    base = np.asarray([-1.0, -2.0, -1.5])
    gain, weights, audit = MODULE.CrossfitBankScorer._mixture_gain(
        base, base, base, base, seed=1)
    assert abs(gain) < 1e-12
    assert np.isclose(np.sum(weights), 1.0)
    assert audit["converged"]


def test_bootstrap_is_deterministic():
    left = MODULE.bootstrap([0.0, 1.0, 2.0], 7, 200)
    right = MODULE.bootstrap([0.0, 1.0, 2.0], 7, 200)
    assert left == right



def test_paired_task_bootstrap_uses_tasks_not_raw_seed_coordinates():
    rows = []
    for task in ("a", "b", "c", "d"):
        for seed in (1, 2):
            rows.append({
                "task": task, "seed": seed,
                "arms": {
                    "cbsi_frontier": {
                        "independent_development_evaluation_gain": 1.25},
                    "cbsi_full": {
                        "independent_development_evaluation_gain": .25},
                },
            })
    difference, interval, task_count = MODULE.paired_task_contrast(
        rows, "cbsi_frontier", "cbsi_full", seed=7, resamples=100)
    assert difference == 1.0
    assert interval == [1.0, 1.0]
    assert task_count == 4


def test_frontier_selection_reuses_positive_descendant_only_on_odd_blocks():
    base = MODULE.FrontierParent("x0", "first", "initial", 0)
    promoted = MODULE.FrontierParent("x0 + x1", "second", "child", 1)
    frontier = [{"parent": promoted, "search_gain": 0.1}]
    assert MODULE.select_frontier_parent(frontier, 0, base) == (base, False)
    assert MODULE.select_frontier_parent(frontier, 1, base) == (promoted, True)
    assert MODULE.select_frontier_parent([], 1, base) == (base, False)


def test_v3_runner_both_new_arms_preserve_budget_and_provenance(monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
        compile_fixed_formula_support,
    )
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        ZERO_PROVIDER_OPERATORS,
    )

    class FakeScorer:
        """Deterministic correctness fixture: not an efficacy experiment."""
        def __init__(self, *args):
            self.score_calls = 0
        def crossfit_gain(self, expression):
            self.score_calls += 1
            return (0.1 + len(expression) * 0.001,
                    0.1 + len(expression) * 0.001)
        def evaluation_gain(self, expression):
            self.score_calls += 1
            return 0.02, [0.5, 0.5], {"converged": True}

    monkeypatch.setattr(MODULE, "CrossfitBankScorer", FakeScorer)
    x = np.column_stack((np.linspace(.2, 1.1, 20),
                         np.linspace(.7, 1.8, 20)))
    parent = SimpleNamespace(
        expression="x0 + x1", structural_hash="archive-parent",
        lineage_id="archive-ancestor")
    archive = SimpleNamespace(candidates=[parent])
    selection = SimpleNamespace(
        development=SimpleNamespace(X=x, y=x[:, 0] + x[:, 1]))
    opportunities = tuple({
        "parent": parent, "donor": "sin(x0)", "site_path": (),
        "opportunity_seed": 101 + idx,
        "opportunity_id": f"opportunity-{idx}",
    } for idx in range(4))
    freeze = {
        "operators": ZERO_PROVIDER_OPERATORS,
        "budget": {
            "edit_evaluations_per_arm": 8,
            "contextual_credit_weight": 0.5,
            "complexity_cost_weight": 0.0,
            "frontier_max_depth": 3,
            "frontier_size": 8,
            "frontier_min_search_gain": 0.0,
        },
    }
    coordinate = {"seed": 2026, "family": "fixture", "task": "synthetic"}
    scores = {}
    for arm in ("cbsi_feasible", "cbsi_frontier"):
        result = MODULE.run_arm(
            arm, freeze, coordinate, archive, [{"expression": "x0"}],
            selection, None, None, None, {}, (x,), opportunities)
        assert result["edit_evaluation_budget"] == 8
        assert result["primary_attempt_count"] == 4
        assert len(result["records"]) == 4
        assert result["provider_calls"] == 0
        assert result["policy"]["schema"] == "cbsi-feasibility-contextual-policy-v1"
        assert all((row["credit"] is None) == (
            not row["counterfactual_pair_valid"]) for row in result["records"])
        scores[arm] = result
    assert not scores["cbsi_feasible"]["frontier_parent_reuses"]
    assert scores["cbsi_feasible"]["shared_opportunities"]
    assert not scores["cbsi_frontier"]["shared_opportunities"]
    assert scores["cbsi_frontier"]["frontier_max_observed_depth"] >= 1
    assert scores["cbsi_frontier"]["frontier_parent_reuses"] >= 1


def test_runner_missing_control_never_becomes_zero_gain(monkeypatch):
    from types import SimpleNamespace
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        StructuralEditingEngine, ZERO_PROVIDER_OPERATORS,
    )

    class FakeScorer:
        def __init__(self, *args):
            self.score_calls = 0
        def crossfit_gain(self, expression):
            self.score_calls += 1
            return (0.4, 0.5)
        def evaluation_gain(self, expression):
            return 0.3, [0.5, 0.5], {"converged": True}

    monkeypatch.setattr(MODULE, "CrossfitBankScorer", FakeScorer)
    def no_control(*args, **kwargs):
        raise ValueError("no matched control available")
    monkeypatch.setattr(StructuralEditingEngine, "matched_control_request",
                        no_control)
    x = np.column_stack((np.linspace(0.2, 1.1, 10),
                         np.linspace(0.4, 1.4, 10)))
    parent = SimpleNamespace(
        expression="x0 + x1", structural_hash="frozen", lineage_id="origin")
    freeze = {
        "operators": ZERO_PROVIDER_OPERATORS,
        "budget": {
            "edit_evaluations_per_arm": 2,
            "complexity_cost_weight": 0.0,
            "contextual_credit_weight": 0.5,
            "frontier_max_depth": 3, "frontier_size": 8,
            "frontier_min_search_gain": 0.0,
        },
    }
    result = MODULE.run_arm(
        "cbsi_feasible", freeze,
        {"seed": 1, "family": "fixture", "task": "synthetic"},
        SimpleNamespace(candidates=[parent]), [{"expression": "x0"}],
        SimpleNamespace(development=SimpleNamespace(X=x)),
        None, None, None, {}, (x,), ({
            "parent": parent, "donor": "sin(x0)",
            "site_path": (), "opportunity_seed": 7,
            "opportunity_id": "frozen-opportunity",
        },))
    assert result["matched_control_count"] == 0
    assert result["records"][0]["credit"] is None
    assert result["records"][0]["control"] is None
    assert not result["policy"]["credit_posterior"][
        result["records"][0]["primary"]["operator"]]["count"]


def test_v3_stage_does_not_inherit_authorization_or_rehash_data(tmp_path):
    import importlib.util
    import sys
    stage_file = SCRIPT.parent / "prepare_cbsi_v3_freeze.py"
    spec = importlib.util.spec_from_file_location("cbsi_stage", stage_file)
    stage_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage_module
    spec.loader.exec_module(stage_module)
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        ZERO_PROVIDER_OPERATORS,
    )
    import hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention as source
    sources = {str(SCRIPT): "old-benchmark-hash",
               str(Path(source.__file__).resolve()): "old-mainline-hash"}
    freeze = {
        "schema": "cbsi-zero-provider-development-freeze-v2",
        "protocol_id": "cbsi-correctness-hotfix-v2",
        "execution_authorized": True,
        "arms": list(stage_module.V3_ARMS[:-2]),
        "operators": list(ZERO_PROVIDER_OPERATORS),
        "file_hashes": sources,
        "budget": {"edit_evaluations_per_arm": 10,
                   "complexity_cost_weight": 0.0},
        "analysis": {"bootstrap_seed": 7, "bootstrap_resamples": 100},
        "artifact_manifest": [{"path": "never-opened", "sha256": "keep-me"}],
        "hdf5_sha256": "keep-hdf5-hash",
    }
    revised = stage_module.stage(freeze)
    assert revised["execution_authorized"] is False
    assert revised["hdf5_sha256"] == "keep-hdf5-hash"
    assert revised["artifact_manifest"] == freeze["artifact_manifest"]
    assert len(revised["arms"]) == 9
    assert revised["protocol_id"] == "cbsi-contextual-frontier-v3"
    assert freeze["file_hashes"] == sources


def test_development_roles_must_be_disjoint():
    from hypothesis_mvp.data.roles import DataRole, RoleDataset
    x = np.column_stack((np.linspace(0, 1, 10),
                         np.linspace(1, 2, 10)))
    left = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0])
    right = RoleDataset(DataRole.VALIDATION, x.copy(), x[:, 0].copy())
    with __import__("pytest").raises(ValueError, match="overlap"):
        MODULE.ensure_development_roles_separate(left, right)


def test_stale_source_provenance_is_rejected():
    with __import__("pytest").raises(ValueError, match="absent"):
        MODULE.verify_source_provenance({})


def test_v3_stage_accepts_unmodified_v1_protocol_as_source():
    import importlib.util
    import sys
    import hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention as source
    path = SCRIPT.parent / "prepare_cbsi_v3_freeze.py"
    spec = importlib.util.spec_from_file_location("cbsi_stage_from_v1", path)
    stage_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage_module
    spec.loader.exec_module(stage_module)
    snapshot = {
        "schema": "cbsi-zero-provider-development-freeze-v1",
        "execution_authorized": True,
        "arms": list(stage_module.V3_ARMS[:-2]),
        "operators": list(source.ZERO_PROVIDER_OPERATORS),
        "file_hashes": {
            str(Path(source.__file__).resolve()): "old-mainline",
            str(SCRIPT): "old-benchmark",
        },
        "budget": {"edit_evaluations_per_arm": 10},
        "analysis": {"bootstrap_seed": 8, "bootstrap_resamples": 200},
        "artifact_manifest": [],
    }
    new = stage_module.stage(snapshot, edit_evaluations_per_arm=32)
    assert new["budget"]["edit_evaluations_per_arm"] == 32
    assert new["v3_budget_change"]["old_edit_evaluations_per_arm"] == 10
    assert new["execution_authorized"] is False
    assert snapshot["budget"]["edit_evaluations_per_arm"] == 10
