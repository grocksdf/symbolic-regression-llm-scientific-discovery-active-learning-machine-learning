"""Production adapter exercised on synthetic correctness data only."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset, AcquisitionCovariates, covariate_fingerprint
from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
from hypothesis_mvp.discovery.proposal_runtime import ProposalBatch, ProposalCandidate
from hypothesis_mvp.discovery.structural_plan_pool import StructuralPlan, plans_from_proposal_batch
from scripts.predictive_plan_search import CONTRACT, MODES, run_predictive_plan_comparison


def setup():
    x = np.column_stack((np.linspace(.3, 1.4, 12), np.linspace(.6, 1.8, 12)**2))
    def data(offset, role=DataRole.VALIDATION, shift=0):
        z = x + offset
        return RoleDataset(role, z, z[:,0]**2 + .2*z[:,1] + shift)
    fit, left, right = data(0, DataRole.DEVELOPMENT), data(.01), data(.02)
    selection = SimpleNamespace(development=fit, validation=left,
        acquisition_pool=AcquisitionCovariates(DataRole.ACQUISITION_POOL, x+.04, covariate_fingerprint(x+.04)))
    ids = {r: tuple(f"{r}:{i}" for i in range(len(x))) for r in ("fit", "left", "right", "cal", "adm", "report")}
    design = ExpansionDesign(ids["fit"]+ids["left"]+ids["right"], ids["cal"], ids["adm"], ids["report"],
                             (0,)*len(x), (0,), tested_mixture_count=2)
    plans = [StructuralPlan(e, "x0", "upper", "complete_edit", "program", f"p{i}")
             for i,e in enumerate(("x0", "x0*x1", "x0**2", "sin(x0)", "log(x0-1)", "exp(x0)"))]
    candidate = ProposalCandidate("c", "island", "x1**2", "PROPOSE_NEW_SKELETON", "correctness", "parent",
                                  lineage_id="llm", prompt_hash="prompt", response_hash="response")
    batch = ProposalBatch((candidate,), True, "ok", "prompt", "response", {"provider_requests": [{
        "provider_all_attempts_preserved": True, "provider_outcomes": [{"provider_http_status": 200}]}]})
    llm, audit = plans_from_proposal_batch(batch, parent_expressions={"parent": "x0"}, region_id="upper")
    kw = dict(implementation_contract=CONTRACT, E0=[{"expression":"x0", "source":"engine", "lineage_id":"core"}],
        plans=tuple(plans)+llm, proposal_audits=[audit], selection=selection, left=left, right=right,
        row_ids={r:ids[r] for r in ("fit","left","right")}, generation_ancestry_ids=ids["fit"],
        design=design, regions=[{"id":"upper","axis":0,"side":"gt","cut":.8}],
        task_identity="synthetic", seed=7, blocks_per_arm=3)
    return kw, ids, data


def execute(root, report_shift=0, admission_shift=0):
    kw, ids, data = setup()
    def calibration():
        assert (root/"GENERATION_FROZEN.json").exists()
        return ids["cal"], data(.05)
    def admission():
        assert all((root/f"{m}_CALIBRATION_FROZEN.json").exists() for m in MODES)
        return ids["adm"], data(.06, shift=admission_shift)
    def reporting():
        assert all((root/f"{m}_ADMISSION_FROZEN.json").exists() for m in MODES)
        return ids["report"], data(.07, shift=report_shift)
    return run_predictive_plan_comparison(**kw, read_calibration=calibration, read_admission=admission,
                                        read_reporting=reporting, output_root=root)


def test_complete_pool_selection_fitting_admission_reporting(tmp_path):
    result = execute(tmp_path/"run")
    assert result["protocol_complete"] and result["feature_dimension"] == 8
    assert result["pool_shared"] and result["candidate_scoring_slots_equal"]
    assert result["shared_proposal_logical_calls"] == 1
    generations = [json.loads((tmp_path/"run"/f"{m}_GENERATION.json").read_text()) for m in MODES]
    assert all(g["executed_slots"] == 6 and len(g["records"]) == 3 for g in generations)
    assert generations[0]["pool_identity"] == generations[1]["pool_identity"]
    statuses = [s["outcome"]["status"] for g in generations for r in g["records"] for s in r["slots"]]
    assert "scored" in statuses and "invalid" in statuses and "duplicate" in statuses
    assert all(g["policy"]["task_identity"] == "synthetic" for g in generations)


def test_reporting_and_admission_cannot_change_search_or_policy(tmp_path):
    execute(tmp_path/"one")
    execute(tmp_path/"two", report_shift=100, admission_shift=-100)
    for name in ["PROPOSAL_POOL_FROZEN.json", "GENERATION_FROZEN.json"] + [f"{m}_GENERATION.json" for m in MODES]:
        assert (tmp_path/"one"/name).read_bytes() == (tmp_path/"two"/name).read_bytes()


def test_unknown_scorer_error_is_terminal_and_cannot_train_zero(monkeypatch, tmp_path):
    import scripts.predictive_plan_search as module
    def broken(*a, **kw): raise FloatingPointError("unresolved numerical failure")
    monkeypatch.setattr(module.DirectCrossfitBankScorer, "crossfit_gain", broken)
    kw, _, _ = setup()
    def closed(): pytest.fail("independent data opened on search failure")
    with pytest.raises(FloatingPointError, match="unresolved"):
        module.run_predictive_plan_comparison(**kw, read_calibration=closed, read_admission=closed,
                                              read_reporting=closed, output_root=tmp_path/"broken")
    assert (tmp_path/"broken"/"FAILURE.json").exists()
    assert not (tmp_path/"broken"/"GENERATION_FROZEN.json").exists()


def test_misbound_llm_provenance_or_ancestry_rejected_before_output(tmp_path):
    kw, ids, _ = setup()
    def closed(): pytest.fail("must stay closed")
    for patch in ({"proposal_audits": []}, {"generation_ancestry_ids":ids["adm"]},
                  {"design":replace(kw["design"], tested_mixture_count=1)}):
        with pytest.raises(ValueError):
            run_predictive_plan_comparison(**{**kw,**patch}, read_calibration=closed, read_admission=closed,
                                          read_reporting=closed, output_root=tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_negative_rewards_are_not_clamped_or_screened_before_independent_admission(monkeypatch, tmp_path):
    import scripts.predictive_plan_search as module
    monkeypatch.setattr(module.DirectCrossfitBankScorer, "crossfit_gain", lambda *a: (-.5, -.2))
    execute(tmp_path/"negative")
    for mode in MODES:
        g = json.loads((tmp_path/"negative"/f"{mode}_GENERATION.json").read_text())
        scored = [s["outcome"] for r in g["records"] for s in r["slots"] if s["outcome"]["status"]=="scored"]
        assert scored and len(g["candidates"]) == len(scored)
        assert all(o["fold_gains"]==[-.5,-.2] for o in scored)
