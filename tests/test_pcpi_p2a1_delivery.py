from __future__ import annotations

import json
from pathlib import Path

from scripts.progress import ProgressReporter
from scripts.run_pcpi_p2a_real import _load_frozen_config, build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_only_the_p2a1_robust_config_is_production_visible() -> None:
    config_path = ROOT / "configs" / "p2a_1_robust_smc.json"
    config = _load_frozen_config(config_path.resolve(), ROOT.resolve())
    assert config["stage"] == "P2A.1"
    assert config["cess_target_fraction"] == 0.8
    assert config["heldout_state"] == "closed"
    assert not (ROOT / "configs" / "p2a_real_smc.json").exists()


def test_real_runner_accepts_only_the_p2a1_phase() -> None:
    parsed = build_parser().parse_args(
        [
            "--data-root", "data",
            "--output-dir", "outputs/run",
            "--source-artifact", "source.zip",
            "--config", "configs/p2a_1_robust_smc.json",
            "--phase", "P2A.1",
            "--heldout-state", "closed",
        ]
    )
    assert parsed.phase == "P2A.1"


def test_progress_is_printed_and_appended_as_jsonl(tmp_path, capsys) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    reporter = ProgressReporter(path)
    reporter.emit("smc_run_started", "run 1/72", completed_runs=0, total_runs=72)
    reporter.emit("smc_run_completed", "run 1/72 complete", completed_runs=1, total_runs=72)
    printed = capsys.readouterr().out
    assert "run 1/72" in printed
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["event"] for item in records] == [
        "smc_run_started",
        "smc_run_completed",
    ]
    assert records[-1]["completed_runs"] == 1
