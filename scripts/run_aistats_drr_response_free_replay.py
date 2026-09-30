"""Short response-free replay of repaired DRR over frozen candidate artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.hypothesis_mvp_pcpi import drr_adapter
from methods.hypothesis_mvp_pcpi._mainline import ensure_mainline

ensure_mainline()
from hypothesis_mvp.hypotheses.source_identity import verify_clean_git_source


FAMILIES = ("bio_pop_growth", "chem_react", "matsci")
CONDITIONS = (
    "full_scientist_v6",
    "entropy_portfolio_v5",
    "no_llm_v6",
    "single_engine_v6",
)


def _sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _take(dataset, indices, columns):
    order = sorted(enumerate(indices), key=lambda row: row[1])
    values = np.asarray(
        dataset[[index for _, index in order], columns], dtype=float)
    restored = np.empty_like(values)
    for sorted_index, (original_index, _) in enumerate(order):
        restored[original_index] = values[sorted_index]
    return restored


def _strict_roles(hdf5, family, task, seed):
    path = f"lsr_synth/{family}/{task}/train"
    if path not in hdf5:
        raise KeyError(f"registered train dataset missing: {path}")
    dataset = hdf5[path]
    count, columns = dataset.shape
    if count < 40 or columns < 2:
        raise ValueError("registered DRR train dataset is infeasible")
    order = drr_adapter._ordered_indices(count, task, seed)
    cuts = (
        int(round(.50 * count)),
        int(round(.70 * count)),
        int(round(.90 * count)),
    )
    indices = {
        "discovery_development": tuple(order[:cuts[0]]),
        "discovery_validation": tuple(order[cuts[0]:cuts[1]]),
        "inference_initial": tuple(order[cuts[1]:cuts[2]]),
        "action_covariates": tuple(order[cuts[2]:]),
    }

    def opened(role):
        rows = _take(dataset, indices[role], slice(None))
        return rows[:, 1:], rows[:, 0]

    development = opened("discovery_development")
    validation = opened("discovery_validation")
    initial = opened("inference_initial")
    actions = _take(dataset, indices["action_covariates"], slice(1, None))
    return drr_adapter.DRRSelectionRoles(
        development[0], development[1],
        validation[0], validation[1],
        initial[0], initial[1],
        actions, indices,
    )


def _artifact(source, family, task, seed, condition):
    generation = (
        "full_scientist_v6"
        if condition == "entropy_portfolio_v5" else condition)
    return (
        source / "runs" / family / task / f"seed{seed}" / generation
        / "pcpi_artifacts" / f"{task}.json"
    )


def _selection_method(condition):
    return (
        drr_adapter._bank.PORTFOLIO_CAPACITY_METHOD
        if condition == "entropy_portfolio_v5"
        else drr_adapter._bank.DECISION_RISK_CAPACITY_METHOD
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-freeze", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("response-free replay output must be new")
    freeze = json.loads(
        args.benchmark_freeze.read_text(encoding="utf-8"))
    if freeze.get("selected_task_count") != 30:
        raise ValueError("unexpected frozen task manifest")
    selected = {
        family: freeze["selected_tasks"][family][0]
        for family in FAMILIES
    }
    seeds = (0, 1, 2)
    hdf5_path = args.data_root / "lsr_bench_data.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    rows = []
    with h5py.File(hdf5_path, "r") as hdf5:
        for family in FAMILIES:
            task = selected[family]
            for seed in seeds:
                roles = _strict_roles(hdf5, family, task, seed)
                for condition in CONDITIONS:
                    artifact = _artifact(
                        args.source_output, family, task, seed, condition)
                    print(
                        f"replay {condition} {family}/{task} seed={seed}",
                        flush=True)
                    if not artifact.is_file():
                        rows.append({
                            "family": family, "task": task, "seed": seed,
                            "condition": condition, "indicator": 0,
                            "ready": False, "status": "missing-artifact",
                            "error": "historical-candidate-artifact-missing",
                        })
                        continue
                    payload = json.loads(
                        artifact.read_text(encoding="utf-8"))
                    report = payload["scientific_discovery_runtime"]
                    candidates = drr_adapter.candidate_rows(
                        report, report["best_expression"],
                        roles.X_development.shape[1])
                    result = drr_adapter.evaluate_drr_candidates(
                        candidates, roles, condition=condition,
                        task_name=task, seed=seed,
                        selection_method=_selection_method(condition))
                    rows.append({
                        "family": family, "task": task, "seed": seed,
                        "condition": condition,
                        "indicator": result.indicator,
                        "ready": result.ready, "status": "replayed",
                        "error": str(
                            result.certificate.get("error_message") or ""),
                        "certificate": result.certificate,
                        "candidate_count": len(candidates),
                    })

    replayed = [row for row in rows if row["status"] == "replayed"]
    indicators = Counter(
        (row["condition"], int(row["indicator"])) for row in rows)
    errors = Counter(
        row["error"] for row in replayed if row["error"])
    core_contract_errors = sum(
        count for message, count in errors.items()
        if "capacity bank preserves registered provenance" in message)
    positive = sum(row["indicator"] for row in replayed)
    decisions = {
        "all_registered_rows_accounted_for": len(rows) == 36,
        "at_least_two_thirds_of_rows_replayed":
            len(replayed) >= 24,
        "protected_core_contract_has_no_error":
            core_contract_errors == 0,
        "repaired_drr_is_not_constant_zero":
            positive > 0,
        "both_readiness_outcomes_observed":
            0 < positive < len(replayed),
        "candidate_responses_stayed_closed": all(
            row.get("certificate", {}).get(
                "candidate_response_accessed", False) is False
            for row in replayed),
        "test_and_ood_stayed_closed": all(
            row.get("certificate", {}).get(
                "test_or_ood_accessed", False) is False
            for row in replayed),
    }
    result = {
        "schema": "scientific-aistats-drr-response-free-replay-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "selection_rule": (
            "first frozen selected task in each non-LSR family; all three "
            "frozen seeds; all four frozen conditions"),
        "selected_tasks": selected,
        "row_count": len(rows),
        "replayed_row_count": len(replayed),
        "missing_artifact_count": len(rows) - len(replayed),
        "positive_indicator_count": positive,
        "indicator_counts": {
            f"{condition}/{indicator}": count
            for (condition, indicator), count in sorted(indicators.items())
        },
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
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "measurement_authorized": False,
        "efficacy_demonstrated": False,
        "claim_boundary": (
            "Historical-candidate response-free DRR functionality screen "
            "only; not corrected primary analysis, efficacy, superiority, "
            "task selection, or rerun authorization."),
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "AISTATS_DRR_RESPONSE_FREE_REPLAY.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items() if key != "rows"
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
