from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

from hypothesis_mvp.hypotheses import (
    delivery_source_tree_hash,
    production_code_hash,
    resolve_formal_source_identity,
    verify_clean_git_source,
    verify_source_artifact,
)
from hypothesis_mvp.pcpi.reference import FIXTURE_ROLE, correctness_diagnostic_bank
from scripts.plot_pcpi_p2b_diagnostic import make_p2b_figure
from scripts.create_delivery_manifest import _inventory, _production_code_hash
from scripts.run_pcpi_p2b_diagnostic import (
    CLAIM_BOUNDARY,
    _load_config,
    build_parser,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p2b_transdimensional_diagnostic.json"


def _write_delivery_fixture(root: Path, archive: Path, *, tamper_archive: bool = False) -> str:
    payload = root / "payload.txt"
    payload.write_bytes(b"registered\n")
    file_hash = sha256(payload.read_bytes()).hexdigest()
    rows = [{"path": "payload.txt", "sha256": file_hash, "size_bytes": payload.stat().st_size}]
    tree = sha256()
    tree.update(b"payload.txt")
    tree.update(b"\0")
    tree.update(bytes.fromhex(file_hash))
    tree_hash = tree.hexdigest()
    manifest = {
        "schema": "pcpi-delivery-manifest-v1",
        "source_tree_sha256": tree_hash,
        "files": rows,
    }
    (root / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with ZipFile(archive, "w") as handle:
        handle.writestr("bundle/DELIVERY_MANIFEST.json", json.dumps(manifest))
        handle.writestr(
            "bundle/payload.txt",
            b"tampered\n" if tamper_archive else b"registered\n",
        )
    return tree_hash


def test_frozen_p2b_config_has_formal_counts_and_fixture_role() -> None:
    config = _load_config(CONFIG, ROOT)
    assert config["particle_counts"] == [128, 512, 2048]
    assert len(config["seeds"]) == 8
    assert config["fixture_role"] == FIXTURE_ROLE
    assert config["heldout_state"] == "not-applicable"


def test_p2b_cli_exposes_no_data_or_heldout_opening_path() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "data_root" not in destinations
    heldout = next(action for action in parser._actions if action.dest == "heldout_state")
    assert heldout.choices == ("not-applicable",)


def test_p2b_claim_is_explicitly_diagnostic_not_efficacy() -> None:
    assert "not real-data efficacy evidence" in CLAIM_BOUNDARY
    assert "held-out confirmation" in CLAIM_BOUNDARY
    assert FIXTURE_ROLE == "inference_correctness_diagnostic_fixture"


def test_source_identity_excludes_outputs_and_delivery_metadata(tmp_path: Path) -> None:
    (tmp_path / "hypothesis_mvp").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "hypothesis_mvp" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "outputs" / "secret.txt").write_text("first", encoding="utf-8")
    manifest = {"schema": "pcpi-delivery-manifest-v1", "source_tree_sha256": "a" * 64}
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = production_code_hash(tmp_path)
    assert _production_code_hash(tmp_path) == before
    (tmp_path / "outputs" / "secret.txt").write_text("changed", encoding="utf-8")
    manifest["source_tree_sha256"] = "b" * 64
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert production_code_hash(tmp_path) == before
    assert delivery_source_tree_hash(tmp_path) == "b" * 64


def test_formal_source_identity_matches_archive_manifest_and_local_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    expected = _write_delivery_fixture(tmp_path, archive)
    assert verify_source_artifact(tmp_path, archive) == expected
    (tmp_path / "payload.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file (size|hash) mismatch"):
        verify_source_artifact(tmp_path, archive)


def test_formal_source_identity_rejects_tampered_archive(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _write_delivery_fixture(tmp_path, archive, tamper_archive=True)
    with pytest.raises(ValueError, match="archive file (size|hash) mismatch"):
        verify_source_artifact(tmp_path, archive)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_direct_source_identity_requires_and_records_a_clean_git_tree(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "pcpi-test@example.invalid")
    _git(tmp_path, "config", "user.name", "PCPI Test")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "fixture")

    identity = verify_clean_git_source(tmp_path)
    assert identity["source_identity_kind"] == "clean_git_worktree"
    assert identity["source_package_hash"] is None
    assert len(str(identity["source_tree_hash"])) == 64
    assert identity["source_git_dirty"] is False
    assert identity["source_tracked_file_count"] == 1
    assert resolve_formal_source_identity(tmp_path) == identity

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worktree is not clean"):
        verify_clean_git_source(tmp_path)


def test_direct_source_identity_rejects_untracked_files(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "pcpi-test@example.invalid")
    _git(tmp_path, "config", "user.name", "PCPI Test")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "fixture")
    (tmp_path / "untracked.txt").write_text("not audited\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worktree is not clean"):
        resolve_formal_source_identity(tmp_path)


def test_declared_mutable_config_can_differ_from_archive_placeholder(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"registered\n")
    local_config = tmp_path / "config" / "provider.json"
    local_config.parent.mkdir()
    local_config.write_text('{"api_key": "local-secret"}\n', encoding="utf-8")
    file_hash = sha256(payload.read_bytes()).hexdigest()
    rows = [{"path": "payload.txt", "sha256": file_hash, "size_bytes": payload.stat().st_size}]
    tree = sha256()
    tree.update(b"payload.txt")
    tree.update(b"\0")
    tree.update(bytes.fromhex(file_hash))
    manifest = {
        "schema": "pcpi-delivery-manifest-v1",
        "source_tree_sha256": tree.hexdigest(),
        "mutable_unregistered_paths": ["config/provider.json"],
        "files": rows,
    }
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with ZipFile(archive, "w") as handle:
        handle.writestr("bundle/DELIVERY_MANIFEST.json", json.dumps(manifest))
        handle.writestr("bundle/payload.txt", b"registered\n")
        handle.writestr("bundle/config/provider.json", '{"api_key": "placeholder"}\n')
    assert verify_source_artifact(tmp_path, archive) == tree.hexdigest()


def test_archive_only_installer_is_hashed_in_zip_but_not_required_to_match_locally(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"registered\n")
    (tmp_path / "install.ps1").write_bytes(b"local prior installer\n")
    file_hash = sha256(payload.read_bytes()).hexdigest()
    installer_bytes = b"canonical archive installer\n"
    installer_hash = sha256(installer_bytes).hexdigest()
    rows = [{"path": "payload.txt", "sha256": file_hash, "size_bytes": payload.stat().st_size}]
    tree = sha256()
    tree.update(b"payload.txt")
    tree.update(b"\0")
    tree.update(bytes.fromhex(file_hash))
    manifest = {
        "schema": "pcpi-delivery-manifest-v1",
        "source_tree_sha256": tree.hexdigest(),
        "archive_only_files": [
            {
                "path": "install.ps1",
                "sha256": installer_hash,
                "size_bytes": len(installer_bytes),
            }
        ],
        "files": rows,
    }
    (tmp_path / "DELIVERY_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with ZipFile(archive, "w") as handle:
        handle.writestr("bundle/DELIVERY_MANIFEST.json", json.dumps(manifest))
        handle.writestr("bundle/payload.txt", b"registered\n")
        handle.writestr("bundle/install.ps1", installer_bytes)
    assert verify_source_artifact(tmp_path, archive) == tree.hexdigest()


def test_delivery_inventory_rejects_cache_artifacts(tmp_path: Path) -> None:
    cache = tmp_path / ".pytest_cache" / "v" / "cache"
    cache.parent.mkdir(parents=True)
    cache.write_text("must not ship\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden delivery path"):
        _inventory(tmp_path, tmp_path / "DELIVERY_MANIFEST.json")


def test_p2b_bank_is_dimension_varying_and_finite() -> None:
    bank = correctness_diagnostic_bank()
    dimensions = {len(item.basis_terms) for item in bank.structures}
    assert dimensions == {1, 2, 3, 4}
    assert len(bank.structures) == 7


def test_p2b_plot_writes_pdf_svg_and_high_resolution_png(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    fields = [
        "particle_count", "structure_tv", "predictive_nll_error",
        "final_unique_root_ancestors", "birth_acceptance_rate",
        "death_acceptance_rate", "replace_acceptance_rate",
        "genealogy_consistent", "resampling_decisions_valid",
        "root_ancestry_monotone",
    ]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for count in (128, 512, 2048):
            for seed in range(2):
                writer.writerow({
                    "particle_count": count,
                    "structure_tv": 0.1 / (seed + 1),
                    "predictive_nll_error": 0.02,
                    "final_unique_root_ancestors": count / 2,
                    "birth_acceptance_rate": 0.4,
                    "death_acceptance_rate": 0.3,
                    "replace_acceptance_rate": 0.5,
                    "genealogy_consistent": True,
                    "resampling_decisions_valid": True,
                    "root_ancestry_monotone": True,
                })
    paths = make_p2b_figure(metrics, tmp_path / "figures")
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    assert all(path.stat().st_size > 1000 for path in paths)
