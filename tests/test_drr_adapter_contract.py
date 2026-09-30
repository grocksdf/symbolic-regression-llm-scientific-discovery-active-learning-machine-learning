"""Train-only DRR benchmark adapter contract fixtures."""

import inspect

import numpy as np

from methods.hypothesis_mvp_pcpi import drr_adapter


def test_drr_split_never_exports_action_targets():
    samples = np.column_stack((
        np.arange(80.), np.arange(80.), np.arange(80.) ** 2))
    roles = drr_adapter.split_training_samples(
        samples, task_name="fixture", seed=3)
    assert roles.X_actions.shape[1] == 2
    assert not hasattr(roles, "y_actions")
    sets = [set(rows) for rows in roles.role_row_indices.values()]
    assert all(not left & right for i, left in enumerate(sets)
               for right in sets[i + 1:])


def test_drr_adapter_public_signature_has_no_test_or_ood_surface():
    parameters = inspect.signature(
        drr_adapter.evaluate_drr_candidates).parameters
    assert not any(token in name.lower() for name in parameters
                   for token in ("test", "ood", "heldout", "action_y"))


def test_candidate_pool_recovers_distinct_core_before_optional_sources():
    report = {
        "final_topk": [
            {"expression": "x0 + x1", "source": "engine:mcts",
             "origin": "deterministic"},
            {"expression": "x0*x1", "source": "llm_evidence_synthesis",
             "origin": "llm"},
        ],
        "evaluated_hypothesis_bank": [
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic"},
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic"},
            {"expression": "x0 + x1", "source": "engine:mcts",
             "origin": "deterministic"},
            {"expression": "x0*x1", "source": "llm_evidence_synthesis",
             "origin": "llm"},
        ],
    }
    rows = drr_adapter.candidate_rows(report, "x0", 2)
    families = [
        drr_adapter._stacking.source_family(row) for row in rows]
    assert families[:2] == ["core", "core"]
    assert {"engine:mcts", "llm"} <= set(families)
    assert len(rows) <= 8


def test_protected_backbone_support_overrides_optional_engine_family():
    report = {
        "final_topk": [],
        "evaluated_hypothesis_bank": [
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic"},
            {"expression": "x0 + x1", "source": "engine:mcts",
             "origin": "deterministic"},
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic"},
        ],
    }
    rows = drr_adapter.candidate_rows(
        report, "x0", 2, protected_expressions=(
            {"expression": "x0 + x1", "engine": "mcts"},))
    protected = next(
        row for row in rows if row["expression"] == "x0 + x1")
    assert protected["source"] == (
        "protected_counterfactual_backbone:engine:mcts")
    assert drr_adapter._stacking.source_family(protected) == "core"


def test_prefix_adapter_has_fixed_response_free_surface():
    samples = np.column_stack((
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200) ** 2))
    roles = drr_adapter.split_training_samples(
        samples, task_name="prefix-fixture", seed=4)
    candidates = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
        {"expression": "x0**2", "source": "deterministic_linear_anchor",
         "origin": "deterministic"},
    ]
    curve = drr_adapter.evaluate_drr_prefix_candidates(
        candidates, roles, condition="fixture",
        task_name="prefix-fixture", seed=4)
    assert curve.prefixes == (8, 16, 32)
    assert curve.to_dict()["action_response_accessed"] is False


def test_operational_qd_replaces_optional_topk(monkeypatch):
    samples = np.column_stack((
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200) ** 2))
    roles = drr_adapter.split_training_samples(
        samples, task_name="qd-fixture", seed=5)
    report = {
        "evaluated_hypothesis_bank": [
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic", "metrics": {"val_nmse": .5}},
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic", "metrics": {"val_nmse": .7}},
            {"expression": "x0+x1", "source": "engine:mcts",
             "origin": "deterministic", "metrics": {"val_nmse": .4}},
        ],
        "final_topk": [],
    }
    seen = {}
    monkeypatch.setattr(
        drr_adapter._qd, "build_operational_qd_repertoire",
        lambda candidates, core, initial, actions, **kwargs: (
            (next(row for row in candidates
                  if row["source"] == "engine:mcts"),),
            seen.setdefault("audit", {"schema": "fixture-qd"})))
    curve = type("Curve", (), {
        "normalized_lower_bounds": (0.1, 0.1, 0.1),
        "normalized_aulc": 0.1})()
    monkeypatch.setattr(
        drr_adapter, "evaluate_drr_prefix_candidates",
        lambda candidates, roles, **kwargs: curve)
    rows, audit = drr_adapter.candidate_rows_operational_qd(
        report, "x0", roles, task_name="qd-fixture", seed=5)
    assert audit["schema"] == "fixture-qd"
    assert audit["conservative_handover"][
        "selected_repertoire"] == "legacy"


def test_operational_qd_handover_requires_prefix_noninferiority(
        monkeypatch):
    samples = np.column_stack((
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200),
        np.linspace(-1., 1., 200) ** 2))
    roles = drr_adapter.split_training_samples(
        samples, task_name="qd-handover", seed=6)
    report = {
        "evaluated_hypothesis_bank": [
            {"expression": "x0", "source": "engine:polynomial_lasso",
             "origin": "deterministic", "metrics": {"val_nmse": .5}},
            {"expression": "1", "source": "deterministic_constant_anchor",
             "origin": "deterministic", "metrics": {"val_nmse": .7}},
            {"expression": "x0+x1", "source": "engine:mcts",
             "origin": "deterministic", "metrics": {"val_nmse": .4}},
        ], "final_topk": []}
    monkeypatch.setattr(
        drr_adapter._qd, "build_operational_qd_repertoire",
        lambda candidates, core, initial, actions, **kwargs: (
            (next(row for row in candidates
                  if row["source"] == "engine:mcts"),),
            {"schema": "fixture-qd"}))
    curves = iter([
        type("Curve", (), {
            "normalized_lower_bounds": (.1, .2, .3),
            "normalized_aulc": .2})(),
        type("Curve", (), {
            "normalized_lower_bounds": (.1, .1, .5),
            "normalized_aulc": .3})(),
    ])
    monkeypatch.setattr(
        drr_adapter, "evaluate_drr_prefix_candidates",
        lambda *args, **kwargs: next(curves))
    _, audit = drr_adapter.candidate_rows_operational_qd(
        report, "x0", roles, task_name="qd-handover", seed=6)
    assert audit["conservative_handover"][
        "targeted_handover"] is False
