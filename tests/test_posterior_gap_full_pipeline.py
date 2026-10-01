"""Controlled inference and source-flow correctness, never efficacy evidence."""

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import FrozenAxisRegions
from hypothesis_mvp.discovery.pcpi_adapter import freeze_discovery_model, freeze_discovery_target
from hypothesis_mvp.discovery.posterior_gap_diagnosis import diagnose_frozen_bank
from hypothesis_mvp.discovery.posterior_gap_evidence import screen_independent_adequacy
from hypothesis_mvp.discovery.regional_candidate_admission import admit_regional_candidates
from hypothesis_mvp.discovery.proposal_runtime import ProposalContext, ProposalRuntime
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.data.roles import AcquisitionCovariates, SelectionData, covariate_fingerprint
from hypothesis_mvp.discovery.agent import DiscoveryAgentConfig
from hypothesis_mvp.discovery.agent import DiscoveryAgent
from hypothesis_mvp.discovery.proposal_runtime import ProviderSettings, ProviderRoute
from hypothesis_mvp.discovery.knowledge_runtime import KnowledgeRuntime


def _roles():
    x = np.linspace(-2., 2., 18)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0] ** 2)
    selected = RoleDataset(DataRole.VALIDATION,
                           np.array([[-2.5], [2.5]]), np.array([6.25, 6.25]))
    gap_x = np.linspace(5., 7., 24)[:, None]
    gap = RoleDataset(DataRole.VALIDATION, gap_x, gap_x[:, 0] ** 2)
    admit_x = np.linspace(3., 4., 24)[:, None]
    admit = RoleDataset(DataRole.VALIDATION, admit_x, admit_x[:, 0] ** 2)
    return fit, selected, gap, admit


def _frozen_bank(fit):
    prior = NormalInverseGammaPrior()
    engine = [{"expression": "x0", "source": "engine:a"},
              {"expression": "x0**3", "source": "engine:b"}]
    model = freeze_discovery_model(engine, n_features=1, prior=prior,
        exploration_identity="a" * 64,
        coefficient_policy="discard-fitted-coefficients-refit-closed-basis")
    domain = np.linspace(-4., 7., 56)[:, None]
    target = freeze_discovery_target(model, fit, domain,
        measurement_budget=4, expected_model_identity=model.stable_hash)
    regions = FrozenAxisRegions(1, 0, (0.,))
    diagnosis = diagnose_frozen_bank(model, target, domain,
        np.full(len(domain), 1. / len(domain)), regions)
    return prior, engine, domain, model, target, regions, diagnosis


def test_independent_undercoverage_unblocks_search_but_not_admission():
    fit, selection, gap, admit = _roles()
    prior, core, domain, model, target, regions, diagnosis = _frozen_bank(fit)
    evidence = screen_independent_adequacy(
        model, target, diagnosis, regions, gap,
        discovery_development=fit, discovery_validation=selection)
    assert evidence.eligible_regions == (1,)
    brief = evidence.prompt_brief(diagnosis)
    assert brief["propose_allowed"] is True
    assert brief["decision_utility_gain_checked"] is False
    rows, report = admit_regional_candidates(
        core, [{"expression": "x0**2", "source": "llm", "origin": "llm"}],
        fit, gap, admit, domain, regions, evidence.eligible_regions,
        prior, 4)
    assert report["attempts"] == 1
    assert report["admission_identity"] == admit.fingerprint
    assert report["decision_contribution_assessed"] is False
    assert report["candidates"][0]["region_selector_mass"][1] > .5
    assert rows and rows[0]["expression"] == "x0**2"


def test_reused_selection_or_admission_rows_fail_before_screening():
    fit, selection, gap, admit = _roles()
    prior, core, domain, model, target, regions, diagnosis = _frozen_bank(fit)
    with pytest.raises(ValueError, match="disjoint"):
        screen_independent_adequacy(model, target, diagnosis, regions,
            selection, discovery_development=fit,
            discovery_validation=selection)
    with pytest.raises(ValueError, match="disjoint"):
        admit_regional_candidates(core, [{"expression": "x0**2", "source": "llm"}],
                                  fit, gap, gap, domain, regions, (1,), prior, 4)


def test_gap_conditioned_typed_correction_enters_existing_proposal_parser():
    runtime = ProposalRuntime(EquationRuntime(2), 2, None, 1)
    context = ProposalContext(1, "balanced", "parent", 2, 2,
        incumbent_expression="x0+x1", existing_supports=(("x0",),),
        gap_directed=True)
    item = {"candidate_id": "correction-a", "parent_hash": "parent",
            "action": "CHANGE_INTERACTION", "equation": "x0",
            "correction": "y_hat*x0", "rationale": "testable interaction"}
    candidate, projection = runtime._candidate(item, 0, context, "p", "r")
    assert projection["materialized_correction"] is True
    assert "y_hat" not in candidate.equation
    assert "x0" in candidate.equation
    with pytest.raises(ValueError, match="duplicates_frozen_bank"):
        runtime._candidate(item, 0, ProposalContext(
            1, "balanced", "parent", 2, 2,
            incumbent_expression="x0+x1",
            existing_supports=(("x0_sq", "x0_x1"),),
            gap_directed=True), "p", "r")
    with pytest.raises(ValueError, match="correction_requires"):
        runtime._candidate(item, 0, ProposalContext(
            1, "balanced", "parent", 2, 2), "p", "r")


def test_quality_first_full_preserves_engine_frontier_and_adds_llm_budget(
        tmp_path, monkeypatch):
    import hypothesis_mvp.discovery.system_ablation as ablation
    fit, selection_data, gap, admit = _roles()
    pool = np.linspace(-4., 7., 56)[:, None]
    selection = SelectionData(fit, selection_data,
        AcquisitionCovariates(DataRole.ACQUISITION_POOL, pool,
                              covariate_fingerprint(pool)), ())
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso", "mcts"), engine_workers=1,
        engine_retries=0, cycles=1, discovery_budget=48,
        engine_budget=2, engine_repeats=1,
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True, posterior_gap_directed=True,
        synthesis_evaluation_reserve=3, llm_evaluation_reserve=4,
        refit_policy="pcpi-closed-basis-amplitudes")
    settings = ProviderSettings(routes=(ProviderRoute(
        "https://example.invalid", "fixture", "fixture"),))
    budgets = {}
    raw = [{"expression": "x0", "engine": "polynomial_lasso", "lineage_id": "a"},
           {"expression": "x0**3", "engine": "mcts", "lineage_id": "b"}]
    def fake_variant(root, variant, variant_config, provider, *args):
        budgets[variant] = variant_config.discovery_budget
        candidates = ([{"expression": "x0**2", "source": "llm",
                       "origin": "llm"}] if variant == "full" else [])
        return {"variant": variant, "provider_calls": 0,
                "provider_attempts_used": 0,
                "posterior_gap": [], "candidates": candidates,
                "dataset": "controlled", "seed": 42,
                "status": "succeeded", "best_val_nmse": 1.,
                "measurement_budget": 0,
                "engine_job_budget": 2,
                "candidate_evaluation_budget": variant_config.discovery_budget,
                "compute_ceiling": 10.,
                "heldout_opened": False, "selection_used_heldout": False,
                "development_fingerprint": fit.fingerprint,
                "validation_fingerprint": selection_data.fingerprint,
                "hypothesis_provenance": {"raw_engine_candidates": raw}}
    monkeypatch.setattr(ablation, "_run_scientist_variant", fake_variant)
    monkeypatch.setattr(ablation, "_screen_gap_candidates", lambda *args: None)
    outcome = ablation._run_scientist_ablations(
        tmp_path, selection, "controlled", config, settings,
        "polynomial_lasso", 10., 3, "fixture", {
            "task_name": "controlled", "task_description": "diagnostic"},
        2, gap, NormalInverseGammaPrior(), 4, admit)
    assert budgets["full"] == budgets["single_engine"] == 55
    assert budgets["no_llm"] == 48
    assert outcome["rows"][0]["hypothesis_provenance"][
        "quality_first_engine_identity"] == outcome["rows"][1][
            "hypothesis_provenance"]["quality_first_engine_identity"]
    assert outcome["rows"][0]["hypothesis_provenance"][
        "intact_engine_support_count"] == 2
    assert outcome["pair_gate"]["shared_engine_frontier"] is True
    assert outcome["pair_gate"]["augmentation_total_per_run"] == 7


def test_existing_memory_records_admission_without_premature_promotion(tmp_path):
    memory = KnowledgeRuntime(tmp_path / "structure_library.jsonl",
                              tmp_path / "runtime_ledger.jsonl")
    memory.staging.append({"stage_id": "stage", "status": "staged",
                           "posterior_gap_mode": True,
                           "final_lineage_id": "candidate-lineage", "entries": []})
    recorded = memory.finalize_gap_stage(
        "stage", admission_identity="separate-admission-role",
        target_identity="common-target",
        candidate_records=({"lineage_id": "candidate-lineage",
                            "admitted": True, "region_selector_mass": [.1, .9],
                            "composed_expression": "x0**2",
                            "attempted_candidate_count": 2,
                            "region_identity": "frozen-regions"},))
    assert recorded["status"] == "prediction_admitted_pending_decision"
    assert recorded["decision_effect_assessed"] is False
    assert recorded["candidate_admission"]["composed_expression"] == "x0**2"
    assert recorded["candidate_admission"]["attempted_candidate_count"] == 2
    assert recorded["candidate_admission"]["region_identity"] == "frozen-regions"
    assert memory.retrieve_task_local((), 8) == []
    assert memory.library_size() == 0


def test_no_independent_gap_abstains_before_scientist_review(monkeypatch):
    from types import SimpleNamespace
    config = DiscoveryAgentConfig(
        engines=("polynomial_lasso", "mcts"), engine_budget=2,
        engine_repeats=1, discovery_budget=55,
        scientist_orchestration=True, typed_evidence_synthesis=True,
        typed_inner_augmentation=True, posterior_gap_directed=True,
        synthesis_evaluation_reserve=3, llm_evaluation_reserve=4)
    agent = DiscoveryAgent(config, ProviderSettings(routes=(ProviderRoute(
        "https://example.invalid", "fixture", "fixture"),)))
    monkeypatch.setattr(agent, "_run_engines", lambda *args:
                        SimpleNamespace(all_results=()))
    monkeypatch.setattr(agent, "_independent_gap_brief", lambda *args: {
        "prompt": {"propose_allowed": False, "regions": []},
        "existing_supports": [], "audit": {"target_identity": "fixture"}})
    monkeypatch.setattr(agent, "_resolve_review", lambda *args:
                        pytest.fail("review provider called despite independent abstention"))
    planner = SimpleNamespace(call_count=0, attempt_count=0, errors=[])
    _, review, _, _, audit, usage = agent._orchestrate_cycle(
        None, 0, planner, {"scientist_state": {}},
        object(), NormalInverseGammaPrior(), 4)
    assert review.stop_reason == "gap-directed proposal abstained"
    assert audit["posterior_gap_brief"]["propose_allowed"] is False
    assert usage[0] == 0
