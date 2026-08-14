from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_installer_uses_canonical_audit_and_preserves_local_assets() -> None:
    text = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "audit_final_source.py" in text
    assert "scripts.verify_delivery" in text
    assert "audit_production_integrity.py" not in text
    assert "File]::Replace" not in text
    assert '"contracts"' in text
    assert '$existingProvider' in text
    assert "$managedDirectories" in text
    assert '"docs"' in text
    assert '"data\\manifests"' in text
    assert '"data\\split_manifests"' in text
    assert 'Install-ManagedPath -RelativePath "outputs"' not in text
    assert "pip install" not in text
    assert "Installed canonical PCPI P2A.1" not in text


def test_source_packager_preserves_the_declared_stage_argument() -> None:
    text = (ROOT / "scripts" / "package_source.ps1").read_text(encoding="utf-8")
    assert "$stagingRoot" in text
    assert "--stage $Stage" in text
    assert "$stage = [system.io.path]::getfullpath" not in text.lower()
