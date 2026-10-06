import copy
from itertools import product
import json

import numpy as np
import pytest

from scripts import run_matched_attribution_matrix as matrix
from scripts.attribution_claim_contract import (
    CONTRASTS, FREEZE_SCHEMA, LEGACY_FREEZE_SCHEMA, NOMINAL_SCOPE,
    primary_pass_conditions, protocol_analysis_fields, statistical_contract,
    task_bootstrap_intervals, validate_registered_freeze, verify_protocol_binding,
)


def frozen(*, method="percentile", resamples=2048, seed=17, tasks=2, seeds=(1, 2)):
    return {"schema": FREEZE_SCHEMA, "claim_scope": NOMINAL_SCOPE,
            "execution_authorized": True, "formal_dependency_end_to_end_gate_verified": True,
            "complete_experiment_protocol": True, "historical_no_go_preserved": True,
            "statistical_analysis": statistical_contract(method=method, resamples=resamples, seed=seed),
            "primary_pass_conditions": primary_pass_conditions(),
            "selected_tasks": {"fixture": [f"T{i}" for i in range(tasks)]},
            "seeds": list(seeds), "coordinate_count": tasks * len(seeds)}


def rows(freeze, l_s=(.1, .2), l_e=(.2, .3), feedback=True):
    return [{"family": "fixture", "task": task, "seed": seed,
             "effects": {"L_minus_S_log_score": l_s[index],
                         "L_minus_E_log_score": l_e[index],
                         "S_minus_E_log_score": l_e[index] - l_s[index]},
             "L_feedback_verified": bool(feedback and index == 0),
             "S_feedback_verified": False}
            for index, task in enumerate(freeze["selected_tasks"]["fixture"])
            for seed in freeze["seeds"]]


def test_positive_means_with_zero_crossing_interval_is_scientific_no_go():
    freeze = frozen()
    result = matrix.summarize(freeze, rows(freeze, l_s=(-.1, .2)))
    assert result["task_mean_effects"]["L_minus_S_log_score"] > 0
    assert result["task_bootstrap"]["intervals"]["L_minus_S_log_score"][0] < 0
    assert result["scientific_verdict"] == "NO-GO"
    assert result["passed"] is False
    assert result["nominal_slot_matched_predictive_effect_authorized"] is False
    assert result["matched_predictive_attribution_authorized"] is False


def test_positive_nominal_effect_cannot_authorize_equal_effective_attribution():
    freeze = frozen()
    result = matrix.summarize(freeze, rows(freeze))
    assert result["passed"] is True
    assert result["nominal_slot_matched_predictive_effect_authorized"] is True
    assert result["equal_effective_opportunities_established"] is False
    assert result["equal_effective_opportunity_attribution_authorized"] is False
    assert result["matched_predictive_attribution_authorized"] is False


def test_exact_zero_lower_bound_cannot_pass_by_epsilon_or_non_strict_rule():
    freeze = frozen()
    result = matrix.summarize(freeze, rows(freeze, l_s=(0., .2)))
    assert result["task_bootstrap"]["intervals"]["L_minus_S_log_score"][0] == 0
    assert result["passed"] is False


def test_second_primary_contrast_and_verified_feedback_are_required():
    freeze = frozen()
    assert not matrix.summarize(freeze, rows(freeze, l_e=(-.1, .2)))["passed"]
    assert not matrix.summarize(freeze, rows(freeze, feedback=False))["passed"]


def test_enumerated_two_task_sign_cases_match_exact_bootstrap_law():
    freeze = frozen(resamples=1024)
    cases = 0
    for values in product((-.1, 0., .2), repeat=4):
        for feedback in (False, True):
            # With two tasks, both endpoint resamples have probability 1/4,
            # so each exact 2.5% percentile is the smaller task contrast.
            exact_lower_s, exact_lower_e = min(values[:2]), min(values[2:])
            expected = feedback and exact_lower_s > 0 and exact_lower_e > 0
            result = matrix.summarize(freeze, rows(
                freeze, l_s=values[:2], l_e=values[2:], feedback=feedback))
            assert result["passed"] is expected
            assert not result["equal_effective_opportunity_attribution_authorized"]
            assert not result["matched_predictive_attribution_authorized"]
            cases += 1
    assert cases == len(list(product((-.1, 0., .2), repeat=4))) * 2


@pytest.mark.parametrize("method", ["percentile", "basic"])
def test_task_bootstrap_matches_independent_pcg_draw_and_quantile_reference(method):
    contract = statistical_contract(method=method, resamples=2048, seed=17)
    effects = {f"fixture/T{i}": dict(zip(CONTRASTS, values)) for i, values in enumerate(
        ((-.1, .2, .3), (.2, .4, .2), (.4, .7, .3)))}
    result = task_bootstrap_intervals(effects, contract)
    values = np.asarray([list(effects[task].values()) for task in sorted(effects)]).T
    indices = np.random.Generator(np.random.PCG64(17)).integers(0, 3, size=(2048, 3))
    draws = np.mean(values[:, indices], axis=-1)
    lower, upper = np.quantile(draws, [.025, .975], axis=1, method="linear")
    if method == "basic":
        lower, upper = 2 * values.mean(axis=1) - upper, 2 * values.mean(axis=1) - lower
    for index, contrast in enumerate(CONTRASTS):
        assert result["intervals"][contrast] == pytest.approx([lower[index], upper[index]])


def test_seeds_are_averaged_within_task_instead_of_becoming_bootstrap_units():
    small, large = frozen(seeds=(1,)), frozen(seeds=tuple(range(20)))
    a = matrix.summarize(small, rows(small))["task_bootstrap"]
    b = matrix.summarize(large, rows(large))["task_bootstrap"]
    assert a == b
    assert not a["seed_rows_used_as_independent_replicates"]


def test_input_order_cannot_change_ci_or_claim():
    freeze = frozen()
    original = rows(freeze)
    assert matrix.summarize(freeze, original) == {
        **matrix.summarize(freeze, list(reversed(original))), "rows": original}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_coordinate_accounting_checks_identities_not_only_row_count(mutation):
    freeze = frozen()
    values = rows(freeze)
    if mutation == "missing":
        values.pop()
    elif mutation == "duplicate":
        values[-1] = copy.deepcopy(values[0])
    else:
        values[-1]["task"] = "unregistered"
    with pytest.raises(ValueError):
        matrix.summarize(freeze, values)


@pytest.mark.parametrize("effect", [float("nan"), float("inf"), True])
def test_nonfinite_and_boolean_effects_fail_closed(effect):
    freeze = frozen()
    values = rows(freeze)
    values[0]["effects"]["L_minus_S_log_score"] = effect
    with pytest.raises(ValueError):
        matrix.summarize(freeze, values)


def test_undefined_bca_primary_interval_is_no_go_without_method_fallback():
    freeze = frozen(method="BCa")
    result = matrix.summarize(freeze, rows(freeze, l_s=(.2, .2), l_e=(.3, .3)))
    assert result["passed"] is False
    assert result["task_bootstrap"]["contract"]["method"] == "BCa"
    assert result["task_bootstrap"]["intervals"]["L_minus_S_log_score"] == [None, None]
    json.dumps(result, allow_nan=False)


def test_degenerate_exploratory_interval_does_not_change_primary_gate():
    freeze = frozen(method="BCa", tasks=6)
    result = matrix.summarize(freeze, rows(freeze, l_s=(1., 2., 3., 4., 5., 6.),
                                         l_e=(1., 2., 3., 4., 5., 6.)))
    assert result["passed"] is True
    assert result["task_bootstrap"]["finite_intervals"] is True
    assert result["task_bootstrap"]["all_reported_intervals_finite"] is False


def test_single_task_does_not_produce_fake_independent_seed_interval():
    freeze = frozen(tasks=1, seeds=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="at least two"):
        matrix.summarize(freeze, rows(freeze, l_s=(.2,), l_e=(.3,)))


def test_old_mean_screen_is_retained_without_claim_or_run_authority():
    freeze = frozen()
    freeze["schema"] = LEGACY_FREEZE_SCHEMA
    result = matrix.summarize(freeze, rows(freeze))
    assert result["legacy_mean_only_screen_passed"] is True
    assert result["passed"] is False
    assert result["matched_predictive_attribution_authorized"] is False
    with pytest.raises(ValueError, match="legacy mean-only"):
        matrix.verify_freeze(freeze)


@pytest.mark.parametrize("mutation", ["method", "seed", "resamples", "confidence_level"])
def test_missing_or_changed_statistical_contract_rejected_before_execution(mutation):
    freeze = frozen()
    if mutation == "confidence_level":
        freeze["statistical_analysis"][mutation] = .90
    else:
        del freeze["statistical_analysis"][mutation]
    with pytest.raises(ValueError):
        matrix.verify_freeze(freeze)


@pytest.mark.parametrize("field", ["complete_experiment_protocol", "primary_pass_conditions",
                                   "execution_authorized", "formal_dependency_end_to_end_gate_verified"])
def test_incomplete_new_freeze_has_no_statistical_authority(field):
    freeze = frozen()
    del freeze[field]
    with pytest.raises(ValueError):
        validate_registered_freeze(freeze)


def test_equal_effective_scope_or_posthoc_equality_boolean_cannot_unlock_claim():
    freeze = frozen()
    freeze["claim_scope"] = "equal_effective_opportunity_attribution"
    with pytest.raises(ValueError, match="matching design"):
        matrix.summarize(freeze, rows(freeze))
    freeze = frozen()
    freeze["equal_effective_opportunities_established"] = True
    with pytest.raises(ValueError):
        matrix.summarize(freeze, rows(freeze))


def test_freeze_contract_is_bound_to_original_registered_protocol():
    freeze = frozen()
    design = {"families": ["fixture"], "tasks_per_family": 2,
              "seeds": [1, 2], "task_selection_namespace": "fresh-fixture"}
    protocol = {**freeze, "execution_authorized": False, "data_design": design}
    freeze.update(registered_families=design["families"], tasks_per_family=2,
                  task_selection_namespace="fresh-fixture", protocol_sha256="fixture-digest")
    verify_protocol_binding(freeze, protocol, "fixture-digest")
    freeze["statistical_analysis"] = statistical_contract(method="percentile", resamples=2048, seed=18)
    with pytest.raises(ValueError, match="differs"):
        verify_protocol_binding(freeze, protocol, "fixture-digest")


def test_registration_requires_ci_rules_and_complete_fresh_protocol():
    protocol = {**frozen(), "execution_authorized": False}
    assert protocol_analysis_fields(protocol)["primary_pass_conditions"] == primary_pass_conditions()
    protocol["complete_experiment_protocol"] = False
    with pytest.raises(ValueError, match="complete protocol"):
        protocol_analysis_fields(protocol)
    protocol["complete_experiment_protocol"] = True
    del protocol["primary_pass_conditions"]["task_bootstrap_95_ci_lower_bounds_strictly_positive"]
    with pytest.raises(ValueError, match="both task-bootstrap"):
        protocol_analysis_fields(protocol)


@pytest.mark.parametrize("blocker", ["incomplete-protocol", "old-ci-rules",
                                    "fixture-gate", "unexecuted-dependency-gate"])
def test_freeze_builder_blocks_incomplete_readiness_before_data_selection(
        tmp_path, monkeypatch, blocker):
    from scripts import build_matched_attribution_freeze as builder
    from scripts.attribution_claim_contract import STATISTICS_SCHEMA
    import scipy
    root = tmp_path / "mainline"
    root.mkdir()
    files = {name: tmp_path / name for name in (
        "hdf5", "provider-env", "provider-gate", "generator-gate", "composition-gate",
        "correctness-gate", "protocol", "exclusion-manifest", "metadata")}
    for path in files.values():
        path.write_text("fixture only")
    source_b, source_m = {"commit": "fixture-b"}, {"commit": "fixture-m"}
    protocol = {**frozen(), "execution_authorized": False,
                "data_design": {"families": ["phys_osc"], "tasks_per_family": 2,
                                "seeds": [1, 2], "task_selection_namespace": "fresh-fixture"}}
    provider = {"classification": "transport_2xx", "http_status": 200,
                "openai_completion_contract_valid": True, "model": "fixture-model",
                "provider_env_path": str(files["provider-env"].resolve())}
    gate = {"passed": True, "schema": "matched-attribution-response-free-gate-v1",
            "formal_dependency_end_to_end_gate_executed": True,
            "statistical_gate_schema": STATISTICS_SCHEMA,
            "attribution_claim_contract_tests_passed": True,
            "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__},
            "julia_runtime": {}, "benchmark_source": source_b,
            "mainline_source": source_m, "config_sha256": "fixture-config"}
    if blocker == "incomplete-protocol":
        protocol["complete_experiment_protocol"] = False
    elif blocker == "old-ci-rules":
        del protocol["primary_pass_conditions"]["task_bootstrap_95_ci_lower_bounds_strictly_positive"]
    elif blocker == "fixture-gate":
        gate["schema"] = "attribution-claim-correctness-fixture-v1"
    else:
        gate["formal_dependency_end_to_end_gate_executed"] = False
    payloads = {"protocol": protocol, "provider-gate": provider,
                "generator-gate": {"passed": True}, "composition-gate": {"passed": True},
                "correctness-gate": gate, "exclusion-manifest": {"selected_tasks": {"phys_osc": []}}}
    for name, payload in payloads.items():
        files[name].write_text(json.dumps(payload))
    monkeypatch.setattr(builder, "validate_config", lambda _: {"llm_model": "fixture-model"})
    monkeypatch.setattr(builder, "source_identity", lambda *args: (source_b, source_m))
    monkeypatch.setattr(builder, "digest", lambda _: "fixture-config")
    def closed_data(*args, **kwargs):
        raise AssertionError("data opened before readiness was established")
    monkeypatch.setattr(builder.h5py, "File", closed_data)
    args = ["--mainline-root", str(root), "--output-dir", str(tmp_path / "output")]
    for name, path in files.items():
        if name != "metadata":
            args += ["--" + name, str(path)]
    args += ["--metadata", "phys_osc=" + str(files["metadata"])]
    with pytest.raises(ValueError):
        builder.main(args)
    assert not (tmp_path / "output").exists()
