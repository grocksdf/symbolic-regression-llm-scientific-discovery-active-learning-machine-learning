"""Prospective Formula Discovery v2 freeze and runner boundaries."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prospective_runner_direct_import_and_execution_surface():
    script = ROOT / "scripts/run_expanded_formula_discovery_prospective.py"
    source = script.read_text(encoding="utf-8")
    assert "full benchmark adapter source unavailable" not in source
    assert "DESIGN_GATE.json" in source
    assert source.index("DESIGN_GATE.json") < source.index(
        "metadata = _metadata_rows(")
    process = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr


def test_v2_freeze_excludes_matsci_and_keeps_confirmation():
    source = (
        ROOT / "scripts/build_expanded_formula_v2_freeze.py"
    ).read_text(encoding="utf-8")
    assert '"matsci"' in source
    assert "inactive MatSci family unexpectedly has fresh tasks" in source
    assert '"untouched_confirmation_tasks"' in source
    assert '"minimum_paired_bank_contrasts": 3' in source
    assert '"execution_authorized": True' in source
