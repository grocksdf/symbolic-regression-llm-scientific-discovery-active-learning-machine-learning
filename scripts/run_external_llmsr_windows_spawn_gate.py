"""No-data Windows spawn correctness Gate for the LLM-SR sandbox."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-site", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("LLM-SR Windows spawn Gate output must be new")
    sys.path.insert(0, str(args.external_site.resolve()))

    from llmsr.evaluator import LocalSandbox

    source = """
import numpy as np

def execute(inputs):
    return (np.asarray(inputs[0], dtype=float) + 1.0).tolist()
"""
    values = np.asarray([1.0, 2.0, 3.0])
    result, runs_ok = LocalSandbox().run(
        source,
        "execute",
        "execute",
        {"data": [values]},
        "data",
        10,
        get_result_sleep_time=0.05,
    )
    result = np.asarray(result, dtype=float)
    decisions = {
        "windows_spawn_completed": bool(runs_ok),
        "child_result_is_correct": bool(
            np.allclose(result, [2.0, 3.0, 4.0])),
        "benchmark_task_arrays_stayed_closed": True,
    }
    payload = {
        "schema": "scientific-external-llmsr-windows-spawn-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "real_data_accessed": False,
        "scientific_prompt_sent": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic process-spawn correctness only; no benchmark, "
            "efficacy, or superiority claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "EXTERNAL_LLMSR_WINDOWS_SPAWN_GATE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
