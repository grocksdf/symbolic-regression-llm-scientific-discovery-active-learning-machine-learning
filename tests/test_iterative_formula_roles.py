"""Synthetic contract checks; no benchmark responses or provider calls."""

from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import json

import numpy as np
import pytest

from methods.hypothesis_mvp_pcpi.drr_adapter import (
    candidate_rows, iterative_gap_role_indices,
    split_three_arm_training_samples, three_arm_role_indices,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig


def test_two_cycle_roles_partition_only_gap_and_admission():
    samples = np.column_stack((np.zeros(200), np.arange(200, dtype=float)))
    original = split_three_arm_training_samples(
        samples, task_name="synthetic", seed=3).role_row_indices
    assert original == three_arm_role_indices(
        200, task_name="synthetic", seed=3)
    pairs = iterative_gap_role_indices(original, 2)
    assert len(pairs) == 2
    gap = set().union(*(set(a) for a, _ in pairs))
    admission = set().union(*(set(b) for _, b in pairs))
    assert gap == set(original["gap_audit"])
    assert admission == set(original["gap_admission"])
    assert all(set(a).isdisjoint(b) for a, b in pairs)
    assert not (gap | admission) & set(original["reporting"])


def test_iterative_roles_reject_duplicate_or_insufficient_indices():
    roles = {"gap_audit": tuple(range(8)),
             "gap_admission": tuple(range(7, 15))}
    with pytest.raises(ValueError, match="overlap"):
        iterative_gap_role_indices(roles, 2)
    with pytest.raises(ValueError, match="insufficient"):
        iterative_gap_role_indices({"gap_audit": (0, 1),
            "gap_admission": tuple(range(8))}, 2)


def test_expanded_candidate_export_keeps_nonlinear_support():
    report = {"evaluated_hypothesis_bank": [
        {"expression": "x0", "source": "engine:pysr", "origin": "engine"},
        {"expression": "exp(-0.7*x0)", "source": "llm_proposal",
         "origin": "llm"}], "final_topk": []}
    rows = candidate_rows(report, "x0", 1,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    assert {row["expression"] for row in rows} == {"x0", "exp(-0.7*x0)"}


def test_searcher_forwards_two_role_pairs_and_exports_admitted_map(
        monkeypatch, tmp_path):
    import methods.hypothesis_mvp_pcpi.drr_searcher as module
    searcher = object.__new__(module.DRRBenchmarkSearcher)
    searcher.condition = "three_arm_formula_expanded_candidate"
    searcher.agent_config = DiscoveryAgentConfig(
        iterative_posterior_refinement=True, cycles=2,
        dataset_family="synthetic")
    searcher.random_seed = 3
    searcher._task_counter = 0
    searcher.llm_enabled = True
    searcher.llm_api_url = "https://example.invalid"
    searcher.llm_model = "fixture"
    searcher.llm_api_key = "fixture"
    searcher.llm_timeout_s = 1.0
    searcher.llm_thinking_type = ""
    searcher.llm_reasoning_effort = ""
    searcher.llm_do_sample = False
    searcher.output_dir = Path(tmp_path)
    searcher.config = SimpleNamespace(llm_temperature=0.0, llm_max_tokens=100)
    searcher.portfolio_method = "fixture"
    monkeypatch.setattr(searcher, "_input_fields", lambda *args:
                        (["y", "x0"], ["response", "input"], ["", ""]))
    captured = {}
    class Agent:
        def __init__(self, config, provider_settings):
            captured["config"] = config
        def run(self, **kwargs):
            captured["kwargs"] = kwargs
            bank = [{"expression": "x0", "source": "engine:pysr"},
                    {"expression": "exp(-0.7*x0)", "source": "llm_proposal",
                     "origin": "llm"}]
            trace = [{"bank_rows_after": bank,
                      "posterior_map_expression": "exp(-0.7*x0)"}] * 2
            return SimpleNamespace(discovery=SimpleNamespace(
                expression="x0", report={"best_programs": ["x0"]}),
                cycles=(), system_evaluation={
                    "iterative_feedback_trace": trace,
                    "iterative_posterior_refinement_verified": True})
    monkeypatch.setattr(module._agent, "DiscoveryAgent", Agent)
    monkeypatch.setattr(module.ProviderSettings, "from_environment",
                        lambda **kwargs: "fixture-provider")
    monkeypatch.setattr(searcher, "_write_artifact", lambda *args: None)
    monkeypatch.setattr(searcher, "_result", lambda task, expression,
                        report, n_features: (expression, report))
    x = np.arange(200, dtype=float)
    task = SimpleNamespace(name="synthetic", desc="synthetic",
                           samples=np.column_stack((x**2, x)))
    expression, report = searcher.discover(task)[0]
    assert expression == "exp(-0.7*x0)"
    assert report["best_programs"] == [expression]
    assert {row["expression"] for row in report["drr_candidate_rows"]} == {
        "x0", expression}
    pairs = captured["kwargs"]["gap_cycle_roles"]
    assert len(pairs) == 2
    assert all(len(pair) == 2 for pair in pairs)
    assert captured["kwargs"]["gap_audit"] is None
    assert captured["kwargs"]["gap_prior"] is not None
    assert report["paired_bank_predictive_report_available"]
    assert not report["measured_pair_authorized"]


def test_post_generation_reporting_includes_no_admission_task(tmp_path):
    import h5py
    from scripts.audit_iterative_bank_reporting import run
    from hypothesis_mvp.data.roles import (
        AcquisitionCovariates, DataRole, RoleDataset, SelectionData,
        covariate_fingerprint,
    )
    from hypothesis_mvp.discovery.pcpi_adapter import (
        freeze_discovery_target, freeze_expanded_formula_model,
    )
    from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

    x = np.linspace(-2., 2., 200)
    samples = np.column_stack((x**2, x))
    hdf5 = tmp_path / "controlled.h5"
    with h5py.File(hdf5, "w") as handle:
        handle.create_dataset("lsr_synth/matsci/fixture/train", data=samples)
    indices = three_arm_role_indices(200, task_name="fixture", seed=3)
    def dataset(name, role):
        values = samples[list(indices[name])]
        return RoleDataset(role, values[:, 1:], values[:, 0])
    fit = dataset("discovery_development", DataRole.DEVELOPMENT)
    validation = dataset("discovery_validation", DataRole.VALIDATION)
    actions = samples[list(indices["action_covariates"]), 1:]
    selection = SelectionData(fit, validation,
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, actions,
                              covariate_fingerprint(actions)), ())
    rows = [{"expression": "x0", "source": "engine:pysr"},
            {"expression": "x0**2", "source": "engine:polynomial_lasso"}]
    identity = sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    model = freeze_expanded_formula_model(rows, n_features=1,
        prior=NormalInverseGammaPrior(), exploration_identity=identity,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    target = freeze_discovery_target(model, fit, actions,
        measurement_budget=2, expected_model_identity=model.stable_hash)
    cycle_indices = iterative_gap_role_indices(indices, 2)
    trace = [{"core_rows_before": rows, "bank_before": model.stable_hash,
              "posterior_before": target.stable_hash, "bank_rows_after": rows,
              "bank_after": model.stable_hash,
              "posterior_after": target.stable_hash,
              "fit_rows": sorted(fit.row_fingerprints),
              "selection_rows": sorted(validation.row_fingerprints),
              "gap_rows": sorted(RoleDataset(DataRole.VALIDATION,
                  samples[list(a), 1:], samples[list(a), 0]).row_fingerprints),
              "admission_rows": sorted(RoleDataset(DataRole.VALIDATION,
                  samples[list(b), 1:], samples[list(b), 0]).row_fingerprints)}
             for a, b in cycle_indices]
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"task_name": "fixture",
        "scientific_discovery_runtime": {
            "drr_condition": "three_arm_formula_expanded_candidate",
            "drr_role_row_indices": {k: list(v) for k, v in indices.items()},
            "iterative_cycle_role_row_indices": [
                {"gap_audit": list(a), "gap_admission": list(b)}
                for a, b in cycle_indices],
            "scientist_agent_system_evaluation": {
                "iterative_feedback_trace": trace,
                "iterative_feedback_gate": {"passed": False, "problems": [
                    "no_admitted_posterior_update_reached_a_later_gap"]}}}}))
    result = run(artifact, hdf5, "matsci", "fixture", 3)
    assert not result["iterative_feedback_witnessed"]
    assert all(row["no_llm_minus_full_mse"] == 0
               for row in result["cycles"])
    assert not result["measured_action_authorized"]
