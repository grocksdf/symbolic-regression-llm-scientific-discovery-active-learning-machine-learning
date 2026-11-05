"""Synthetic correctness fixtures; no scientific efficacy run."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset, AcquisitionCovariates, covariate_fingerprint
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import CandidateArchive
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
from hypothesis_mvp.discovery.region_site_attribution import FEATURE_NAMES, ASSOCIATION_FEATURES
from scripts.regional_structural_search import CONTRACT, MODES, project_contexts, run_regional_site_comparison


def setup():
    x = np.column_stack((np.linspace(.5, 1.5, 10), np.linspace(.6, 1.8, 10)**2))
    def data(offset, role=DataRole.VALIDATION, y_offset=0):
        z = x + offset
        return RoleDataset(role, z, z[:,0] + .2*z[:,1]**2 + y_offset)
    fit, left, right = data(0, DataRole.DEVELOPMENT), data(.01), data(.02)
    selection = SimpleNamespace(development=fit, validation=left,
        acquisition_pool=AcquisitionCovariates(DataRole.ACQUISITION_POOL, x+.04, covariate_fingerprint(x+.04)))
    core = [{"expression":"x0*x1 + sin(x1)", "source":"engine:pysr", "lineage_id":"e"},
            {"expression":"(-4.5)", "source":"engine:pysr", "lineage_id":"constant"}]
    archive = CandidateArchive.freeze(core, n_features=2, coefficient_policy=EXPANDED_FORMULA_POLICY)
    ids = {r: tuple(f"{r}:{i}" for i in range(10)) for r in ("fit", "left", "right", "cal", "adm", "report")}
    design = ExpansionDesign(ids["fit"]+ids["left"]+ids["right"], ids["cal"], ids["adm"], ids["report"],
                             (0,)*10, (0,), tested_mixture_count=6)
    kwargs = dict(implementation_contract=CONTRACT, E0=core, archive=archive, selection=selection,
        left=left, right=right, row_ids={r:ids[r] for r in ("fit","left","right")},
        generation_ancestry_ids=ids["fit"], design=design, seed=3, task_identity="synthetic",
        blocks_per_arm=2, axes=(1,), operators=("subtree_replacement", "correction_term"))
    return kwargs, ids, data


def execute(root, reporting_offset=0):
    kw, ids, data = setup()
    def calibration():
        assert (root / "GENERATION_FROZEN.json").exists()
        assert all((root / f"{m}_GENERATION.json").exists() for m in MODES)
        return ids["cal"], data(.05)
    def admission():
        assert all((root / f"{m}_CALIBRATION_FROZEN.json").exists() for m in MODES)
        return ids["adm"], data(.06)
    def reporting():
        assert all((root / f"{m}_ADMISSION_FROZEN.json").exists() for m in MODES)
        return ids["report"], data(.07, y_offset=reporting_offset)
    return run_regional_site_comparison(**kw, read_calibration=calibration, read_admission=admission,
                                       read_reporting=reporting, output_root=root)


def test_complete_joint_edits_to_independent_admission(tmp_path):
    root = tmp_path / "run"
    result = execute(root)
    assert result["protocol_complete"] and result["provider_calls"] == 0 and not result["frontier_enabled"]
    audits = [json.loads((root/f"{m}_GENERATION.json").read_text()) for m in MODES]
    assert all(a["materialization_slots"] == 4 for a in audits)
    assert all(len(a["records"]) == 2 for a in audits)
    assert any(a["policy"]["observation_count"] for a in audits)
    assert any(r["learned"] and r["contrast"] == "site" for a in audits for r in a["records"])
    for a in audits:
        assert all(o["task_identity"] == "synthetic" for o in a["policy"]["observations"])
        assert a["policy"]["prior_identity"] is None
    assert (root/"RESULT.json").exists()
    opportunities = json.loads((root/"OPPORTUNITIES.json").read_text())
    pairs = {(r["parent"], r["region"]["id"]) for r in opportunities}
    assert len(opportunities) == 2*len(pairs)
    for parent, region in pairs:
        assert {r["contrast"] for r in opportunities if r["parent"]==parent and r["region"]["id"]==region} == {"site","operator"}


def test_reporting_has_no_effect_on_generation_learning_or_admission(tmp_path):
    execute(tmp_path/"one")
    execute(tmp_path/"two", reporting_offset=100)
    for m in MODES:
        a,b = [json.loads((tmp_path/p/f"{m}_GENERATION.json").read_text()) for p in ("one","two")]
        assert a["records"] == b["records"]
        assert a["policy"] == b["policy"]
        assert (tmp_path/"one"/f"{m}_ADMISSION_FROZEN.json").read_bytes() == (tmp_path/"two"/f"{m}_ADMISSION_FROZEN.json").read_bytes()


def test_shuffling_changes_only_associations_preserves_feature_distribution():
    snapshot = {"sites":[{"region_id":"r", "site_path":[i], "features":[float(i+j) for j in range(len(FEATURE_NAMES))]}
                         for i in range(4)]}
    original = project_contexts(snapshot, "r", "E_attributed_joint", 5)
    shuffled = project_contexts(snapshot, "r", "F_shuffled_joint", 5)
    for j,name in enumerate(FEATURE_NAMES):
        p,q = [r["features"][j] for r in original], [r["features"][j] for r in shuffled]
        if name in ASSOCIATION_FEATURES:
            assert sorted(p)==sorted(q) and all(a!=b for a,b in zip(p,q))
        else:
            assert p==q
    local = project_contexts(snapshot, "r", "D_local_joint", 5)
    assert all(r["features"][FEATURE_NAMES.index("residual_association")] == 0 for r in local)


def test_invalid_ancestry_or_multiplicity_rejected_before_readers(tmp_path):
    kw, ids, data = setup()
    for edit in ({"generation_ancestry_ids":ids["adm"]}, {"design":replace(kw["design"], tested_mixture_count=1)}):
        with pytest.raises(ValueError):
            run_regional_site_comparison(**{**kw,**edit}, read_calibration=lambda:pytest.fail("opened"),
                read_admission=lambda:pytest.fail("opened"), read_reporting=lambda:pytest.fail("opened"),
                output_root=tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_generation_failure_preserves_evidence_and_closes_independent_data(monkeypatch, tmp_path):
    import scripts.run_cbsi_zero_provider_experiment as scorer
    def broken(*a,**k): raise ValueError("scorer contract failed")
    monkeypatch.setattr(scorer.CrossfitBankScorer, "crossfit_gain", broken)
    kw, _, _ = setup()
    with pytest.raises(ValueError, match="scorer contract"):
        run_regional_site_comparison(**kw, read_calibration=lambda:pytest.fail("opened"),
            read_admission=lambda:pytest.fail("opened"), read_reporting=lambda:pytest.fail("opened"), output_root=tmp_path/"broken")
    assert (tmp_path/"broken"/"FAILURE.json").exists()
    assert not (tmp_path/"broken"/"GENERATION_FROZEN.json").exists()
