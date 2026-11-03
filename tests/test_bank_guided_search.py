import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset, AcquisitionCovariates, covariate_fingerprint
from hypothesis_mvp.discovery.joint_evidence_gated_expansion import CandidateArchive
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.search_budget import SearchBudget
from scripts.bank_guided_search import CONTRACT, prepare_generation, run_guided_comparison, generate_arm


def fixture():
    x = np.column_stack((np.linspace(.2, 1.3, 14), np.linspace(.7, 1.8, 14) ** 2))
    def role(offset, kind):
        z = x + offset
        return RoleDataset(kind, z, z[:, 0] + .2 * z[:, 1] ** 2)
    fit = role(0., DataRole.DEVELOPMENT)
    left, right, evaluation = (role(v, DataRole.VALIDATION) for v in (.01, .02, .03))
    selection = SimpleNamespace(development=fit, validation=left,
                                acquisition_pool=AcquisitionCovariates(
                                    DataRole.ACQUISITION_POOL, x + .04, covariate_fingerprint(x + .04)))
    core = [{"expression": "x0", "source": "engine:pysr", "lineage_id": "c0"},
            {"expression": "x1", "source": "engine:pysr", "lineage_id": "c1"}]
    archive = CandidateArchive.freeze(core, n_features=2, coefficient_policy=EXPANDED_FORMULA_POLICY)
    ids = {name: tuple(f"{name}:{i}" for i in range(14)) for name in ("fit", "left", "right", "evaluation")}
    return core, archive, selection, left, right, evaluation, ids


def execute(tmp_path, *, purpose="mechanism", budgets=None, offset=0.):
    core, archive, selection, left, right, evaluation, ids = fixture()
    root = tmp_path / "run"
    def read():
        assert (root / "GENERATION_FROZEN.json").exists()
        for mode in ("bank", "residual", "blind"):
            assert (root / f"{mode}_GENERATION.json").exists()
        return ids["evaluation"], RoleDataset(DataRole.VALIDATION, evaluation.X, evaluation.y + offset)
    result = run_guided_comparison(
        implementation_contract=CONTRACT, purpose=purpose,
        budgets=budgets or dict.fromkeys(("bank", "residual", "blind"), SearchBudget(3)),
        E0=core, archive=archive, selection=selection, left=left, right=right,
        row_ids=ids, read_evaluation=read, seed=7, output_root=root)
    return result, root


def test_actual_generation_to_frozen_evaluation(tmp_path):
    result, root = execute(tmp_path)
    assert result["protocol_complete"] and result["provider_calls"] == 0
    audits = [json.loads((root / f"{mode}_GENERATION.json").read_text()) for mode in ("bank", "residual", "blind")]
    assert all(len(audit["records"]) == 3 for audit in audits)
    parent_sites = [[(r["parent"], r["site_path"]) for r in a["records"]] for a in audits]
    assert parent_sites[0] == parent_sites[1] == parent_sites[2]
    assert any("request" in row for audit in audits for row in audit["records"])
    assert len({audit["bank_identity"] for audit in audits}) == 1
    for audit in audits:
        for key in ("density_evaluations", "optimizer_fits"):
            assert audit["budget"]["work"][key] + audit["scorer_setup_compute"][key] == audit["search_compute"][key]
        assert audit["budget"]["work"]["wall_time_seconds"] > 0
        assert audit["generation_wall_seconds"] >= audit["scorer_setup_wall_seconds"]
    for mode in ("bank", "residual", "blind"):
        assert len((root / f"{mode}_ATTEMPTS.jsonl").read_text().splitlines()) == 3


def test_quality_allows_unequal_predeclared_work(tmp_path):
    _, root = execute(tmp_path, purpose="quality", budgets={
        "bank": SearchBudget(2), "residual": SearchBudget(3), "blind": SearchBudget(4)})
    for mode, count in (("bank", 2), ("residual", 3), ("blind", 4)):
        assert len(json.loads((root / f"{mode}_GENERATION.json").read_text())["records"]) == count


def test_reporting_changes_do_not_change_generated_candidates(tmp_path):
    _, first = execute(tmp_path / "a", offset=0.)
    _, second = execute(tmp_path / "b", offset=100.)
    for mode in ("bank", "residual", "blind"):
        one = json.loads((first / f"{mode}_GENERATION.json").read_text())
        two = json.loads((second / f"{mode}_GENERATION.json").read_text())
        assert one["records"] == two["records"]
        assert one["selection_identity"] == two["selection_identity"]


def test_overlap_and_legacy_contract_rejected_before_reading(tmp_path):
    core, archive, selection, left, right, _, ids = fixture()
    for contract, manifest in (("cbsi-comparator-aware-search-v1", ids),
                               (CONTRACT, {**ids, "evaluation": ids["fit"]})):
        with pytest.raises(ValueError):
            run_guided_comparison(implementation_contract=contract, purpose="mechanism",
                                  budgets=dict.fromkeys(("bank", "residual", "blind"), SearchBudget(2)),
                                  E0=core, archive=archive, selection=selection, left=left, right=right,
                                  row_ids=manifest, read_evaluation=lambda: pytest.fail("opened"),
                                  seed=7, output_root=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_no_reporting_after_generation_failure(monkeypatch, tmp_path):
    import scripts.bank_guided_search as adapter
    def fail(*args, **kwargs):
        raise ValueError("synthetic generation failure")
    monkeypatch.setattr(adapter, "prepare_generation", fail)
    core, archive, selection, left, right, _, ids = fixture()
    with pytest.raises(ValueError, match="synthetic"):
        run_guided_comparison(implementation_contract=CONTRACT, purpose="mechanism",
                              budgets=dict.fromkeys(("bank", "residual", "blind"), SearchBudget(2)),
                              E0=core, archive=archive, selection=selection, left=left, right=right,
                              row_ids=ids, read_evaluation=lambda: pytest.fail("opened"),
                              seed=7, output_root=tmp_path / "failed")
    assert (tmp_path / "failed" / "FAILURE.json").exists()


def test_feedback_actually_changes_donor_requests_without_evaluation(monkeypatch):
    core, archive, selection, left, right, _, ids = fixture()
    feedback, values, parents, _ = prepare_generation(core, archive, selection,
                                                     row_ids=ids["fit"], forbidden_row_ids=ids["evaluation"])
    # Orthogonal artificial directions verify routing only, not efficacy.
    u = selection.development.X[:, 0]
    v = np.cos(np.linspace(0, 6, len(u)))
    feedback = replace(feedback, location_score=tuple(u), residual=tuple(v))
    values = {"x0": u, "sin(x1)": v}
    class Sealed:
        @property
        def X(self):
            pytest.fail("evaluation accessed during generation")
        @property
        def y(self):
            pytest.fail("evaluation accessed during generation")
    audits = []
    for mode in ("bank", "residual"):
        _, _, audit = generate_arm(mode, SearchBudget(1), feedback=feedback,
                                   material_values=values, parents=parents, archive=archive,
                                   E0=core, selection=selection, left=left, right=right,
                                   evaluation=Sealed(), seed=7, namespace="synthetic",
                                   operators=("correction_term",))
        audits.append(audit)
    assert audits[0]["records"][0]["donor"] == "x0"
    assert audits[1]["records"][0]["donor"] == "sin(x1)"


def test_scorer_contract_failure_is_terminal_not_zero_gain(monkeypatch, tmp_path):
    import scripts.run_cbsi_zero_provider_experiment as runner

    def broken_score(self, expression):
        raise ValueError("synthetic scorer contract mismatch")
    monkeypatch.setattr(runner.CrossfitBankScorer, "crossfit_gain", broken_score)
    core, archive, selection, left, right, _, ids = fixture()
    with pytest.raises(ValueError, match="contract mismatch"):
        run_guided_comparison(implementation_contract=CONTRACT, purpose="mechanism",
                              budgets=dict.fromkeys(("bank", "residual", "blind"), SearchBudget(4)),
                              E0=core, archive=archive, selection=selection, left=left, right=right,
                              row_ids=ids, read_evaluation=lambda: pytest.fail("opened"),
                              seed=7, output_root=tmp_path / "contract_failure")
    assert (tmp_path / "contract_failure" / "FAILURE.json").exists()
    assert not (tmp_path / "contract_failure" / "GENERATION_FROZEN.json").exists()
