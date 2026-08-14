import json
from pathlib import Path

import pytest

from scripts.run_rd1_1_real_development import (
    _confirmed_library,
    _contract,
    _paired_effects,
    _variant_args,
    _variants,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "rd1_1_real_development_contract.json"


def test_rd1_contract_seals_heldout_and_matches_engine_jobs() -> None:
    contract = _contract(CONTRACT_PATH)
    budget = contract["full_system_budget"]
    assert contract["heldout_policy"]["opened"] is False
    assert contract["heldout_policy"]["available_to_selection"] is False
    single = _variant_args("single_engine", budget)
    assert single[single.index("--engines") + 1] == "polynomial_lasso"
    assert int(single[single.index("--engine-repeats") + 1]) == budget["engine_jobs_per_cycle"]
    assert "no_acquisition" not in _variants("ablation", None)
    with pytest.raises(ValueError, match="canonical P3B"):
        _variant_args("no_acquisition", budget)
    assert "--no-llm" in _variant_args("no_llm", budget)


def test_unconfirmed_knowledge_is_not_scheduled_or_accepted(tmp_path) -> None:
    assert "knowledge_on" not in _variants("ablation", None)
    library = tmp_path / "structure_library.jsonl"
    manifest = tmp_path / "manifest.json"
    library.write_text('{"entry_id":"unconfirmed"}\n', encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema": "confirmed-source-knowledge-v1",
        "source_only": True,
        "contains_target_heldout_rows": False,
        "source_datasets": ["source"],
        "independent_confirmations": [],
        "library_sha256": "wrong",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no independent confirmations"):
        _confirmed_library(library, manifest)


def test_knowledge_pair_uses_knowledge_on_as_reference() -> None:
    rows = [
        {"dataset": "target", "seed": 42, "variant": "full", "best_val_nmse": 0.5},
        {"dataset": "target", "seed": 42, "variant": "no_llm", "best_val_nmse": 0.6},
        {"dataset": "target", "seed": 42, "variant": "knowledge_on", "best_val_nmse": 0.4},
        {"dataset": "target", "seed": 42, "variant": "knowledge_off", "best_val_nmse": 0.45},
    ]
    effects = _paired_effects(rows)
    indexed = {row["ablation"]: row for row in effects}
    assert indexed["no_llm"]["reference_variant"] == "full"
    assert indexed["knowledge_off"]["reference_variant"] == "knowledge_on"
    assert indexed["knowledge_off"]["ablation_minus_reference_val_nmse"] == pytest.approx(0.05)
