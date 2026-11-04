"""Synthetic numerical and visibility checks, without experimental data."""
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_diagnostic_context import (
    build_diagnostic_snapshot, project_diagnostic_scope, validate_diagnostic_view,
)
from hypothesis_mvp.discovery.bank_search_feedback import (
    _hash, attach_feedback, feedback_from_posterior, install_feedback, validate_bank_prompt_feedback,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def snapshot():
    return build_diagnostic_snapshot(np.arange(8.)[:, None], np.arange(8.) - 5,
                                     np.array([-2.] * 4 + [2.] * 4), axes=(0,))


def view(scope, diagnostic=None):
    return {"mode": "blind" if scope == "none" else "bank",
            **project_diagnostic_scope(diagnostic or snapshot(), scope)}


def test_opposite_regions_survive_zero_global_mean():
    data = snapshot()
    summary = data["global_summary"]
    assert summary["mean_location_score"] == 0
    assert summary["mean_abs_location_score"] == 2
    assert summary["positive_location_fraction"] == summary["negative_location_fraction"] == .5
    assert [r["mean_location_score"] for r in data["regions"]] == [-2, 2]
    joint = view("joint", data)
    assert [r["location_score_minus_global"] for r in joint["global_local_links"]] == [-2, 2]
    assert all(not r["opposite_mean_direction"] for r in joint["global_local_links"])
    validate_diagnostic_view(joint)


def test_region_opposes_nonzero_global_direction_and_reconstructs_summary():
    x = np.arange(6.)[:, None]
    data = build_diagnostic_snapshot(x, [-2.] * 3 + [-1.] * 3, [-2.] * 3 + [4.] * 3, axes=(0,))
    joint = view("joint", data)
    assert data["global_summary"]["mean_location_score"] == 1.
    assert [r["opposite_mean_direction"] for r in joint["global_local_links"]] == [True, False]
    for key in set(data["global_summary"]) - {"count"}:
        reconstructed = sum(r[key] * (r["count"] / 6) for r in data["regions"])
        assert reconstructed == pytest.approx(data["global_summary"][key])
    validate_diagnostic_view(joint)


def test_shared_summary_fields_are_exact_scope_projections():
    joint, local, whole, blind = (view(scope) for scope in ("joint", "local", "global", "none"))
    assert joint["regions"] == local["regions"]
    assert joint["global_summary"] == whole["global_summary"]
    assert local["global_summary"] is None and local["global_local_links"] == []
    assert whole["regions"] == [] and whole["global_local_links"] == []
    assert blind["global_summary"] is None and blind["regions"] == []
    for payload in (joint, local, whole, blind):
        validate_diagnostic_view(payload)
    local["regions"][0]["count"] = 999
    assert joint["regions"][0]["count"] == 4


def test_partitions_depend_only_on_covariates_and_all_nonempty_regions_remain():
    x = np.array([[0., 9.], [0., 9.], [1., 9.], [2., 9.]])
    a = build_diagnostic_snapshot(x, [0, 1, 2, 3], [-2, -1, 1, 2], axes=(1, 0))
    b = build_diagnostic_snapshot(x, [30, 20, 10, 0], [4, 3, 2, 1], axes=(1, 0))
    geometry = lambda data: [(r["axis"], r["side"], r["cut"], r["count"]) for r in data["regions"]]
    assert geometry(a) == geometry(b)
    assert geometry(a) == [(1, "le", 9., 4), (0, "le", .5, 2), (0, "gt", .5, 2)]
    validate_diagnostic_view(view("joint", a))


def test_single_observation_remains_descriptive_and_finite():
    data = build_diagnostic_snapshot([[1.]], [-10.], [0.], axes=(0,))
    assert data["global_summary"]["count"] == 1
    assert len(data["regions"]) == 1
    assert data["global_summary"]["positive_location_fraction"] == 0.
    validate_diagnostic_view(view("joint", data))


def test_extreme_finite_values_are_not_silently_clipped():
    data = build_diagnostic_snapshot([[1e308], [1e308]], [1e308, 1e308], [-1e308, 1e308], axes=(0,))
    assert data["global_summary"]["mean_abs_location_score"] == 1e308
    assert data["regions"][0]["cut"] == 1e308
    validate_diagnostic_view(view("joint", data))


@pytest.mark.parametrize("axes", [None, (), (0, 0), (1,), (-1,), (True,), [0], ([0],)])
def test_no_inferred_or_invalid_partition_design(axes):
    with pytest.raises(ValueError, match="axes"):
        build_diagnostic_snapshot([[1.]], [0.], [1.], axes=axes)


@pytest.mark.parametrize("corruption", ["global_in_local", "local_in_global", "blind_signal", "negative_count",
                                        "bad_link", "false_aggregate", "bad_fraction", "nan", "extra_field"])
def test_invalid_scope_or_inconsistent_evidence_fails_closed(corruption):
    data = view("joint")
    if corruption == "global_in_local":
        data["diagnostic_scope"] = "local"
    elif corruption == "local_in_global":
        data["diagnostic_scope"] = "global"
    elif corruption == "blind_signal":
        data["diagnostic_scope"] = "none"
        data["mode"] = "blind"
    elif corruption == "negative_count":
        data["regions"][0]["count"] = -1
    elif corruption == "bad_link":
        data["global_local_links"][0]["location_score_minus_global"] = 10
    elif corruption == "false_aggregate":
        data["global_summary"]["mean_abs_location_score"] = 20
    elif corruption == "bad_fraction":
        data["regions"][0]["positive_location_fraction"] = 2
    elif corruption == "nan":
        data["regions"][0]["mean_log_density"] = float("nan")
    else:
        data["regions"][0]["reporting_gain"] = 5
    with pytest.raises(ValueError):
        validate_diagnostic_view(data)


def actual_feedback(axes=None, role=DataRole.DEVELOPMENT, x=None):
    if x is None:
        x = np.linspace(-1., 1., 10)[:, None]
    data = RoleDataset(role, x, x[:, 0] ** 2 + .1 * x[:, 0])
    model = freeze_expanded_formula_model(
        [{"expression": "x0", "source": "engine:fixture"}], n_features=x.shape[1],
        prior=NormalInverseGammaPrior(), exploration_identity="f" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY, minimum_supports=1)
    engine = model.engine(model.stable_hash)
    posterior = engine.fit_batch(data.X, data.y)
    feedback = feedback_from_posterior(engine, posterior, data,
        search_row_ids=tuple(f"fit:{i}" for i in range(len(x))), forbidden_row_ids=("independent:0",),
        bank_identity=model.stable_hash, include_bank_inventory=True, diagnostic_axes=axes)
    return data, feedback


def test_actual_multiscale_path_avoids_legacy_unselected_axis_overflow():
    x = np.column_stack((np.linspace(-1., 1., 10), np.full(10, 1e308)))
    with np.errstate(over="raise", invalid="raise"):
        _, feedback = actual_feedback((0,), x=x)
        joint = feedback.prompt_payload("bank", diagnostic_scope="joint")
    assert {r["axis"] for r in joint["regions"]} == {0}
    validate_bank_prompt_feedback(joint)


def test_legacy_payload_and_identity_are_preserved():
    _, legacy = actual_feedback()
    fields = dict(legacy.__dict__)
    fields.pop("diagnostics_json")
    assert legacy.identity == _hash(fields)
    assert legacy.prompt_payload()["schema"] == "bank-complementary-proposal-feedback-v1"
    with pytest.raises(ValueError, match="frozen"):
        legacy.prompt_payload(diagnostic_scope="joint")


def test_actual_posterior_snapshot_projects_same_inventory_and_correct_metrics():
    _, feedback = actual_feedback((0,))
    payloads = [feedback.prompt_payload("blind" if scope == "none" else "bank", diagnostic_scope=scope)
                for scope in ("none", "global", "local", "joint")]
    assert all(validate_bank_prompt_feedback(p) for p in payloads)
    assert all(p["bank_inventory"] == payloads[0]["bank_inventory"] for p in payloads)
    assert payloads[1]["global_summary"] == payloads[3]["global_summary"]
    assert payloads[2]["regions"] == payloads[3]["regions"]
    assert payloads[0]["signal_identity"] is None
    assert payloads[1]["global_summary"]["mean_location_score"] == pytest.approx(np.mean(feedback.location_score))
    context = attach_feedback({"new_skeleton_quota": 1}, feedback, mode="bank", diagnostic_scope="joint")
    assert context["new_skeleton_quota"] == 1


def test_explicit_multiscale_roles_are_fitting_only():
    with pytest.raises(ValueError, match="fitting"):
        actual_feedback((0,), role=DataRole.VALIDATION)


def test_real_runtime_context_carries_registered_scope():
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    _, feedback = actual_feedback((0,))
    runtime = SimpleNamespace(config=SimpleNamespace(feedback_generation_policy="legacy", new_skeleton_quota=1),
        orchestration_context={}, evaluation=SimpleNamespace(validity_attempts=[], failure_signature=lambda *args: ()))
    install_feedback(runtime, feedback, mode="bank", diagnostic_scope="joint")
    context = ScientificDiscoveryRuntime._proposal_context(runtime,
        SimpleNamespace(compact=lambda: {}, dag=SimpleNamespace(expression="x0")),
        SimpleNamespace(as_prompt_dict=lambda: {}), "composition")
    assert context["bank_predictive_search_feedback"]["diagnostic_scope"] == "joint"
    assert context["new_skeleton_quota"] == 1

