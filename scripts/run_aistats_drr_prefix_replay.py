"""Fixed-prefix continuous DRR replay over frozen historical candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline
from scripts.run_aistats_drr_response_free_replay import (
    CONDITIONS, FAMILIES, _artifact, _selection_method, _sha, _strict_roles,
)

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("prefix replay output must be new")
    freeze = json.loads(
        args.benchmark_freeze.read_text(encoding="utf-8"))
    selected = {
        family: freeze["selected_tasks"][family][0]
        for family in FAMILIES
    }
    hdf5_path = args.data_root / "lsr_bench_data.hdf5"
    rows = []
    with h5py.File(hdf5_path, "r") as hdf5:
        for family in FAMILIES:
            task = selected[family]
            for seed in (0, 1, 2):
                roles = _strict_roles(hdf5, family, task, seed)
                for condition in CONDITIONS:
                    artifact = _artifact(
                        args.source_output, family, task, seed, condition)
                    print(
                        f"prefix replay {condition} "
                        f"{family}/{task} seed={seed}", flush=True)
                    if not artifact.is_file():
                        rows.append({
                            "family": family, "task": task, "seed": seed,
                            "condition": condition,
                            "status": "missing-artifact",
                            "normalized_aulc": 0.0,
                            "positive_prefix_count": 0,
                            "error": "historical-candidate-artifact-missing",
                        })
                        continue
                    report = json.loads(
                        artifact.read_text(encoding="utf-8")
                    )["scientific_discovery_runtime"]
                    candidates = drr_adapter.candidate_rows(
                        report, report["best_expression"],
                        roles.X_development.shape[1])
                    try:
                        curve = drr_adapter.evaluate_drr_prefix_candidates(
                            candidates, roles, condition=condition,
                            task_name=task, seed=seed,
                            selection_method=_selection_method(condition))
                        payload = curve.to_dict()
                        errors = sorted({
                            str(item.get("error_message"))
                            for item in payload["certificates"]
                            if item.get("error_message")
                        })
                        rows.append({
                            "family": family, "task": task, "seed": seed,
                            "condition": condition, "status": "replayed",
                            "normalized_aulc": payload["normalized_aulc"],
                            "positive_prefix_count":
                                payload["positive_prefix_count"],
                            "normalized_lower_bounds":
                                payload["normalized_lower_bounds"],
                            "prefixes": payload["prefixes"],
                            "certificate_errors": errors,
                            "error": "",
                        })
                    except Exception as error:
                        rows.append({
                            "family": family, "task": task, "seed": seed,
                            "condition": condition, "status": "replay-error",
                            "normalized_aulc": 0.0,
                            "positive_prefix_count": 0,
                            "error": f"{type(error).__name__}:{error}",
                        })

    replayed = [row for row in rows if row["status"] == "replayed"]
    positive = [row for row in replayed if row["normalized_aulc"] > 0.0]
    errors = Counter(
        message
        for row in replayed
        for message in row.get("certificate_errors", ()))
    values = {round(float(row["normalized_aulc"]), 15) for row in replayed}
    decisions = {
        "all_registered_rows_accounted_for": len(rows) == 36,
        "at_least_two_thirds_of_rows_replayed": len(replayed) >= 24,
        "fixed_prefixes_are_8_16_32": all(
            row.get("prefixes") == [8, 16, 32] for row in replayed),
        "no_outer_replay_exception": not any(
            row["status"] == "replay-error" for row in rows),
        "log_mass_failures_absent": not any(
            marker in message
            for message in errors
            for marker in (
                "math domain error",
                "weights and degrees must be positive",
                "no posterior predictive mass")),
        "continuous_signal_is_not_constant_zero": bool(positive),
        "continuous_signal_varies_across_rows": len(values) >= 2,
    }
    result = {
        "schema": "scientific-aistats-drr-prefix-replay-v1",
        "method": "fixed-prefix-certified-normalized-decision-risk-aulc-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "selected_tasks": selected,
        "row_count": len(rows),
        "replayed_row_count": len(replayed),
        "positive_aulc_count": len(positive),
        "distinct_aulc_count": len(values),
        "certificate_errors": dict(sorted(errors.items())),
        "rows": rows,
        "source_output": str(args.source_output.resolve()),
        "source_result_sha256": _sha(
            args.source_output / "AISTATS_DRR_RESULT.json"),
        "benchmark_freeze_sha256": _sha(args.benchmark_freeze),
        "hdf5_sha256": _sha(hdf5_path),
        "benchmark_source": verify_clean_git_source(ROOT),
        "mainline_source": verify_clean_git_source(
            ROOT.parent / "hypothesis_mvp"),
        "engines_or_llm_ran": False,
        "candidate_response_accessed": False,
        "action_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Historical-candidate fixed-prefix expected-risk functionality "
            "screen only; not realized acquisition efficacy, corrected "
            "primary analysis, task selection, or superiority."),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "AISTATS_DRR_PREFIX_REPLAY.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
