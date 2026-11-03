import json

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
from scripts.frozen_predictive_expansion import FrozenBankComponents, run_comparison


def setup():
    d = ExpansionDesign(tuple(f"fit:{i}" for i in range(10)),
                        tuple(f"cal:{i}" for i in range(8)),
                        tuple(f"adm:{i}" for i in range(8)),
                        tuple(f"report:{i}" for i in range(8)), (0,) * 8, (0,))
    x = np.linspace(-1, 1, 10)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0] ** 2 + .2 * x[:, 0])
    core = [{"expression": "x0", "source": "engine:pysr", "lineage_id": "e0"}]
    optional = [{"expression": "x0**2", "source": "llm:typed", "lineage_id": "c1"}]
    components = FrozenBankComponents(core, optional, fit, fit_row_ids=d.training_ids, design=d)
    return d, components


def reader(ids, offset=0.):
    def read():
        x = np.linspace(-.9, .9, len(ids))[:, None]
        return ids, RoleDataset(DataRole.VALIDATION, x, x[:, 0] ** 2 + offset)
    return read


def test_actual_symbolic_models_freeze_before_label_read(tmp_path):
    d, components = setup()
    root = tmp_path / "run"
    events = []

    def admission():
        assert (root / "CALIBRATION_FROZEN.json").exists()
        assert not (root / "ADMISSION_FROZEN.json").exists()
        events.append("admission")
        return reader(d.admission_ids)()

    def reporting():
        assert (root / "ADMISSION_FROZEN.json").exists()
        events.append("reporting")
        return reader(d.reporting_ids, offset=10.)()

    result = run_comparison(design=d, components=components,
                            read_calibration=reader(d.calibration_ids),
                            read_admission=admission, read_reporting=reporting,
                            output_root=root, seed=17)
    assert events == ["admission", "reporting"]
    assert result["component_density_evaluations"] == 6
    assert result["post_calibration_refit"] is False
    frozen = json.loads((root / "CALIBRATION_FROZEN.json").read_text())
    assert result["mixture_identity"] == frozen["mixture_identity"]
    assert set(result["gains"]) == {"engine", "independent_validation", "predictive_stacking",
                                    "screened_predictive_stacking"}


def test_reporting_change_cannot_change_weights_or_admission(tmp_path):
    snapshots = []
    for index, offset in enumerate((0., 100.)):
        d, components = setup()
        root = tmp_path / str(index)
        run_comparison(design=d, components=components,
                       read_calibration=reader(d.calibration_ids),
                       read_admission=reader(d.admission_ids),
                       read_reporting=reader(d.reporting_ids, offset), output_root=root, seed=11)
        snapshots.append(tuple((root / name).read_bytes() for name in
                               ("CALIBRATION_FROZEN.json", "ADMISSION_FROZEN.json")))
    assert snapshots[0] == snapshots[1]


def test_admission_failure_preserves_artifacts_and_never_reads_reporting(tmp_path):
    d, components = setup()
    root = tmp_path / "run"

    def broken():
        raise RuntimeError("synthetic admission failure")

    def sealed():
        pytest.fail("reporting was opened before admission succeeded")

    with pytest.raises(RuntimeError, match="synthetic"):
        run_comparison(design=d, components=components, read_calibration=reader(d.calibration_ids),
                       read_admission=broken, read_reporting=sealed, output_root=root, seed=11)
    assert (root / "CALIBRATION_FROZEN.json").exists()
    assert (root / "FAILURE.json").exists()
    assert not (root / "RESULT.json").exists()
    with pytest.raises(FileExistsError):
        run_comparison(design=d, components=components, read_calibration=sealed,
                       read_admission=sealed, read_reporting=sealed, output_root=root, seed=11)


def test_wrong_source_rows_rejected_before_model_evaluation(tmp_path):
    d, components = setup()
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="source row"):
        run_comparison(design=d, components=components, read_calibration=reader(d.reporting_ids),
                       read_admission=lambda: pytest.fail("opened"),
                       read_reporting=lambda: pytest.fail("opened"), output_root=root, seed=11)
    assert components.evaluation_count == 0


def test_heldout_role_forbidden():
    _, components = setup()
    with pytest.raises(ValueError, match="validation"):
        components.densities(RoleDataset(DataRole.UNTOUCHED_HELDOUT, [[1.]], [2.]))


def test_changed_posterior_fails_closed():
    d, components = setup()
    # Model identity binds the actual fitted posterior, not a caller's label.
    model, engine, posterior, _ = components._components[0]
    components._components = ((model, engine, posterior, "tampered"), *components._components[1:])
    with pytest.raises(ValueError, match="changed after freeze"):
        components.densities(reader(d.admission_ids)()[1])


def test_write_failure_keeps_reporting_sealed(monkeypatch, tmp_path):
    import scripts.frozen_predictive_expansion as adapter

    original = adapter._publish
    def publish(path, value):
        if path.name == "ADMISSION_FROZEN.json":
            raise OSError("synthetic storage failure")
        original(path, value)
    monkeypatch.setattr(adapter, "_publish", publish)
    d, components = setup()
    with pytest.raises(OSError):
        run_comparison(design=d, components=components, read_calibration=reader(d.calibration_ids),
                       read_admission=reader(d.admission_ids),
                       read_reporting=lambda: pytest.fail("opened after write failure"),
                       output_root=tmp_path / "failed", seed=11)
    assert (tmp_path / "failed" / "FAILURE.json").exists()


def test_zero_optional_candidates_produces_core_only_comparison(tmp_path):
    d, _ = setup()
    x = np.linspace(-1, 1, 10)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0])
    core = [{"expression": "x0", "source": "engine:pysr", "lineage_id": "core"}]
    components = FrozenBankComponents(core, [], fit, fit_row_ids=d.training_ids, design=d)
    result = run_comparison(design=d, components=components,
                            read_calibration=reader(d.calibration_ids),
                            read_admission=reader(d.admission_ids),
                            read_reporting=reader(d.reporting_ids), output_root=tmp_path / "core", seed=11)
    assert result["gains"] == dict.fromkeys(result["gains"], 0.)
    assert result["candidate_count"] == 0
    assert result["mixture_accepted"] is False


def test_duplicate_support_not_hidden_by_metadata():
    d, _ = setup()
    x = np.linspace(-1, 1, 10)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0])
    core = [{"expression": "x0", "source": "engine:pysr", "lineage_id": "core"}]
    duplicate = [{"expression": "x0", "source": "llm:typed", "lineage_id": "new-label"}]
    with pytest.raises(ValueError, match="unique non-core"):
        FrozenBankComponents(core, duplicate, fit, fit_row_ids=d.training_ids, design=d)
