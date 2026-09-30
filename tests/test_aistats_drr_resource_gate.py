"""Static no-data resource-ledger checks."""

from pathlib import Path

from scripts.run_aistats_drr_resource_gate import (
    _lsr_hdf5_identity, _provider_key,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_v5_and_v6_configs_share_generation_controls():
    v5 = (ROOT / "configs/aistats_drr_full_v5.yaml").read_text(
        encoding="utf-8")
    v6 = (ROOT / "configs/aistats_drr_full_v6.yaml").read_text(
        encoding="utf-8")
    assert v5.replace(
        "condition: entropy_portfolio_v5",
        "condition: full_scientist_v6").replace(
        "portfolio_method: v5", "portfolio_method: v6").replace(
        "name: AISTATS-DRR-Full-V5",
        "name: AISTATS-DRR-Full-V6") == v6
    no_llm = (ROOT / "configs/aistats_drr_no_llm_v6.yaml").read_text(
        encoding="utf-8")
    assert "protected_counterfactual_backbone: true" in v6
    assert "protected_counterfactual_backbone: true" in no_llm
    assert "conservative_allocation_policy: {}" in v6


def test_resource_gate_reads_registered_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=fixture\n", encoding="utf-8")
    assert _provider_key(path) == "fixture"


def test_lsr_data_identity_requires_one_hash_bound_file(tmp_path):
    import hashlib
    path = tmp_path / "lsr_bench_data.hdf5"
    path.write_bytes(b"fixture")
    identity = _lsr_hdf5_identity(tmp_path)
    assert identity == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(b"fixture").hexdigest(),
        "size_bytes": 7,
    }
