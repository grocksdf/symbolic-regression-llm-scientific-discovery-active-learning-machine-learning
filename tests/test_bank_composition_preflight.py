"""Synthetic metadata only: no experiment data or provider use."""
import json
from dataclasses import replace
import pytest
from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
from scripts.bank_composition_preflight import validate_generation_ancestry, validate_historical_bank_manifest
from scripts.bank_composition_preflight import main


def test_ancestor_access_is_registered_without_refitting_those_rows():
    design = ExpansionDesign(("fit", "old-validation"), ("cal",), ("adm",), ("report",), (0,), (0,))
    audit = validate_generation_ancestry(design, ("fit",), ("fit", "old-validation"), 1)
    assert audit["fitting_row_ids"] == ("fit",)
    with pytest.raises(ValueError, match="ancestry"):
        validate_generation_ancestry(design, ("fit",), ("fit",), 1)
    with pytest.raises(ValueError, match="overlap"):
        replace(design, calibration_ids=("old-validation",))


def test_manifest_checks_all_dependencies_without_replacing_failed_coordinate(tmp_path):
    good = {"family":"fixture", "task":"task", "seed":1, "coordinate_failure":False,
            "coordinate_root":str(tmp_path)}
    bank = {"shared":{"task":"task", "seed":1, "E0_rows":[{"expression":"x0", "source":"engine:fixture"}]}}
    (tmp_path/"SHARED_BLOCK.json").write_text(json.dumps(bank))
    bad = {"family":"fixture", "task":"failed", "seed":1, "coordinate_failure":True}
    with pytest.raises(ValueError, match='"index": 1'):
        validate_historical_bank_manifest([good,bad])
    assert validate_historical_bank_manifest([good])[0]["core_rows"] == bank["shared"]["E0_rows"]
    bank["shared"]["seed"] = 2
    (tmp_path/"SHARED_BLOCK.json").write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="incomplete"):
        validate_historical_bank_manifest([good])


def test_cli_fails_before_producing_receipt_for_missing_core(tmp_path):
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"coordinates":[{"family":"fixture", "task":"failed",
        "seed":1,"coordinate_failure":True}]}))
    output = tmp_path/"new"/"receipt.json"
    with pytest.raises(ValueError, match="incomplete"):
        main(["--manifest",str(manifest),"--output",str(output)])
    assert not output.parent.exists()
