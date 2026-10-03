"""Aggregation contract for the tiny development materialization smoke."""

import subprocess
import sys
from pathlib import Path

from scripts.run_expanded_formula_v2_tiny_development_smoke import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_materialization_rate_is_conditioned_on_healthy_attempted_runs():
    rows = [
        {"status": "success",
         "provider_health": {"provider_attempted": True,
                             "transport_valid": True},
         "provider_cost": [{"http_status": 200}],
         "typed_directive_count": 1, "optional_materialized_count": 1},
        {"status": "success",
         "provider_health": {"provider_attempted": True,
                             "transport_valid": True},
         "provider_cost": [{"http_status": 200}],
         "typed_directive_count": 1, "optional_materialized_count": 0},
    ]
    result = summarize(rows)
    assert result[
        "materialized_novel_candidate_given_successful_llm_response_run"
    ] == 0.5
    assert all(result["decisions"].values())


def test_formula_child_direct_script_import_surface():
    process = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts/run_aistats_three_arm_formula_child.py"),
         "--help"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
