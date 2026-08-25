"""Run the one-shot P3G.1 calibration gate and conditionally acquisition audit.

This runner uses only provenance-verified measured UCI responses.  Validation
responses are consumed by the preregistered prequential calibration gate.  If
any registered run rejects, acquisition is not executed.  If all runs pass,
the posterior is reset to H0 and the matched-budget acquisition comparison is
run automatically without using validation responses for selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from hypothesis_mvp.data import (
    load_registered_real_dataset,
    prepare_real_pool_oracle,
    prepare_real_selection,
)
from hypothesis_mvp.hypotheses import (
    production_code_hash,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from hypothesis_mvp.pcpi.acquisition import (
    ClassPartition,
    PredictiveComponents,
    categorical_entropy,
    class_partition,
    estimate_class_eig_until_ranked,
    predictive_variance,
)
from hypothesis_mvp.pcpi.reference import (
    DevelopmentStandardizer,
    DiscrepancyKernelState,
    RegisteredStructurewiseDiscrepancyEngine,
    SequentialReferencePosterior,
    StructurewiseDiscrepancyPrior,
    aggregate_operational_classes,
    budget_resolved_distance_threshold,
    fit_bank_preconditioner,
    generic_real_bank,
    pit_e_process,
    stable_budget_indices,
)


CONFIG_SCHEMA = "pcpi-p3g1-structurewise-discrepancy-calibration-gate-v1"
PCPI_POLICY = "pcpi_nuisance_aware_joint_eig"
POLICIES = ("random", "uncertainty", "qbc", PCPI_POLICY)


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _purpose_seed(seed: int, purpose: str) -> int:
    digest = sha256(f"pcpi-p3g1:{purpose}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _response_order(row_ids: np.ndarray, seed: int) -> np.ndarray:
    identifiers = np.asarray(row_ids, dtype=object).reshape(-1)
    keys = np.asarray(
        [
            sha256(f"pcpi-p3g1:validation-order:{seed}:{item}".encode()).digest()
            for item in identifiers
        ],
        dtype="|S32",
    )
    return np.argsort(keys, kind="stable")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, str]:
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("P3G.1 requires a clean tracked worktree")
    values = {}
    for name, arguments in {
        "commit": ("rev-parse", "HEAD"),
        "tree": ("rev-parse", "HEAD^{tree}"),
    }.items():
        values[name] = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return values


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("schema") != CONFIG_SCHEMA
        or tuple(config.get("datasets", ()))
        != ("uci_ccpp", "uci_gas_turbine_co", "uci_gas_turbine_nox")
        or tuple(config.get("seeds", ())) != tuple(range(2026080701, 2026080709))
        or tuple(config.get("policies", ())) != POLICIES
        or config.get("predictive_calibration", {}).get("familywise_false_alarm_level") != "1/100"
        or config.get("predictive_calibration", {}).get("equal_per_run_false_alarm_level") != "1/2400"
        or config.get("authorization", {}).get("unconditional_acquisition_execution") is not False
        or config.get("authorization", {}).get("conditional_acquisition_execution")
        != "only-after-all-24-calibration-runs-pass"
        or config.get("authorization", {}).get("candidate_responses")
        != "sealed-until-global-calibration-pass-then-selected-one-at-a-time"
        or config.get("authorization", {}).get("heldout") is not False
    ):
        raise ValueError("P3G.1 frozen protocol was modified")
    return config


@dataclass(frozen=True)
class RunContext:
    dataset_id: str
    seed: int
    standardizer: DevelopmentStandardizer
    engine: RegisteredStructurewiseDiscrepancyEngine
    initial_rows: tuple[int, ...]
    initial_y: np.ndarray
    validation_rows: tuple[int, ...]
    validation_y: np.ndarray
    validation_order: np.ndarray
    candidate_rows: tuple[int, ...]
    candidate_pool_indices: np.ndarray
    structural_assignment: tuple[int, ...]
    structural_class_ids: tuple[str, ...]
    engine_hash: str


def _structure_designs(bank, preconditioner, domain_X) -> dict[str, np.ndarray]:
    return {
        structure.structure_id: preconditioner.transform(
            domain_X, structure.basis_terms
        )
        for structure in bank.structures
    }


def _make_context(frame, config: dict[str, Any], seed: int) -> RunContext:
    prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
    selection = prepared.selection
    initial_indices = stable_budget_indices(
        prepared.development_row_ids,
        int(config["initial_observation_budget"]),
        _purpose_seed(seed, "initial"),
    )
    validation_indices = stable_budget_indices(
        prepared.validation_row_ids,
        int(config["validation_budget"]),
        _purpose_seed(seed, "validation"),
    )
    candidate_indices = stable_budget_indices(
        prepared.acquisition_pool_row_ids,
        int(config["candidate_pool_budget"]),
        _purpose_seed(seed, "candidate"),
    )
    standardizer = DevelopmentStandardizer.fit(
        selection.development.X[initial_indices], selection.development.y[initial_indices]
    )
    initial_X = standardizer.transform_X(selection.development.X[initial_indices])
    validation_X = standardizer.transform_X(selection.validation.X[validation_indices])
    candidate_X = standardizer.transform_X(selection.acquisition_pool.X[candidate_indices])
    initial_y = standardizer.transform_y(selection.development.y[initial_indices])
    validation_y = standardizer.transform_y(selection.validation.y[validation_indices])
    domain_X = np.vstack((initial_X, validation_X, candidate_X))
    bank = generic_real_bank(domain_X.shape[1])
    preconditioner = fit_bank_preconditioner(bank, initial_X)
    posterior_config = config["posterior"]
    kernels = tuple(DiscrepancyKernelState(**item) for item in posterior_config["kernel_states"])
    prior = StructurewiseDiscrepancyPrior(
        posterior_config["discrepancy_probability"],
        posterior_config["discrepancy_precision"],
    )
    engine = RegisteredStructurewiseDiscrepancyEngine(
        bank,
        domain_X,
        kernels,
        prior,
        structure_designs=_structure_designs(bank, preconditioner, domain_X),
        maximum_discrepancy_rank=int(posterior_config["maximum_discrepancy_rank"]),
    )
    nominal = SequentialReferencePosterior(bank, 1.0, preconditioner)
    nominal_posterior = nominal.fit_batch(initial_X, initial_y)
    threshold = budget_resolved_distance_threshold(
        int(config["acquisition_observation_budget"])
    )
    classes = aggregate_operational_classes(
        nominal, nominal_posterior, validation_X, distance_threshold=threshold
    )
    partition = class_partition(nominal_posterior, classes)
    offset = len(initial_X)
    candidate_offset = offset + len(validation_X)
    return RunContext(
        dataset_id=frame.dataset_id,
        seed=seed,
        standardizer=standardizer,
        engine=engine,
        initial_rows=tuple(range(len(initial_X))),
        initial_y=initial_y,
        validation_rows=tuple(range(offset, candidate_offset)),
        validation_y=validation_y,
        validation_order=_response_order(prepared.validation_row_ids[validation_indices], seed),
        candidate_rows=tuple(range(candidate_offset, candidate_offset + len(candidate_X))),
        candidate_pool_indices=candidate_indices,
        structural_assignment=partition.structure_to_class,
        structural_class_ids=partition.class_ids,
        engine_hash=engine.stable_hash,
    )


def _initial_state(context: RunContext):
    state = context.engine.prior_state()
    for row, target in zip(context.initial_rows, context.initial_y, strict=True):
        state = context.engine.update(state, row, float(target))
    return state


def _calibration_run(context: RunContext, config: dict[str, Any]) -> dict[str, Any]:
    state = _initial_state(context)
    pits = []
    for order_index in context.validation_order:
        row = context.validation_rows[int(order_index)]
        target = float(context.validation_y[int(order_index)])
        law = context.engine.sequential_predictive_law(state, (row,))
        pit = float(law.cdf(np.asarray([target]))[0])
        clip = float(config["predictive_calibration"]["pit_clip"])
        pits.append(float(np.clip(pit, clip, 1.0 - clip)))
        state = context.engine.update(state, row, target)
    process = pit_e_process(
        pits,
        false_alarm_level=float(
            Fraction(
                config["predictive_calibration"]["equal_per_run_false_alarm_level"]
            )
        ),
    )
    return {
        "dataset_id": context.dataset_id,
        "seed": context.seed,
        "engine_hash": context.engine_hash,
        "pit_count": len(pits),
        "pit_final_e_value": process.final_e_value,
        "pit_maximum_e_value": process.maximum_e_value,
        "pit_first_rejection_round": process.first_rejection_round,
        "pit_rejected": process.rejected,
        "heldout_opened": False,
        "candidate_responses_used": False,
    }


def _structural_probabilities(context: RunContext, state) -> np.ndarray:
    count = len(context.structural_class_ids)
    result = np.zeros(count, dtype=float)
    structure_lookup = {
        structure.structure_id: index
        for index, structure in enumerate(context.engine.bank.structures)
    }
    for probability, record in zip(state.probabilities, context.engine.records, strict=True):
        structure_index = structure_lookup[record["structure"].structure_id]
        result[context.structural_assignment[structure_index]] += probability
    return result


def _joint_partition(context: RunContext, law) -> ClassPartition:
    structure_lookup = {
        structure.structure_id: index
        for index, structure in enumerate(context.engine.bank.structures)
    }
    labels = []
    for structure_id, record in zip(law.structure_ids, context.engine.records, strict=True):
        class_index = context.structural_assignment[structure_lookup[structure_id]]
        labels.append((class_index, str(record["kernel"])))
    unique = tuple(sorted(set(labels)))
    groups = tuple(tuple(i for i, label in enumerate(labels) if label == item) for item in unique)
    probabilities = tuple(float(np.sum(law.probabilities[np.asarray(group)])) for group in groups)
    assignment = tuple(unique.index(label) for label in labels)
    return ClassPartition(
        tuple(f"structural-class-{item[0]}|nuisance-{item[1]}" for item in unique),
        groups,
        probabilities,
        assignment,
    )


def _components(context: RunContext, state, rows: tuple[int, ...]) -> PredictiveComponents:
    law = context.engine.sequential_predictive_law(state, rows)
    return PredictiveComponents(
        law.probabilities,
        law.degrees_freedom,
        law.locations,
        law.scales,
        _joint_partition(context, law),
    )


def _random_scores(seed: int, round_index: int, count: int) -> np.ndarray:
    return np.random.default_rng(_purpose_seed(seed, f"random:{round_index}")).random(count)


def _policy_scores(context, state, rows, policy: str, round_index: int, config):
    components = _components(context, state, rows)
    if policy == "random":
        return _random_scores(context.seed, round_index, len(rows)), False
    if policy == "uncertainty":
        return predictive_variance(components), False
    if policy == "qbc":
        rng = np.random.default_rng(_purpose_seed(context.seed, f"qbc:{round_index}"))
        draws = rng.choice(
            len(components.structure_probabilities),
            size=int(config["qbc_committee_size"]),
            p=components.structure_probabilities,
        )
        return np.var(components.locations[draws], axis=0, ddof=1), False
    estimate = estimate_class_eig_until_ranked(
        components,
        int(config["eig_quadrature_min_evaluations"]),
        int(config["eig_quadrature_max_evaluations"]),
        error_safety_factor=float(config["eig_quadrature_error_safety_factor"]),
        growth_factor=int(config["eig_quadrature_growth_factor"]),
    )
    if estimate.ranking_certified:
        return estimate.estimate.scores, True
    return _random_scores(context.seed, round_index, len(rows)), False


def _validation_rmse(context: RunContext, state) -> float:
    law = context.engine.sequential_predictive_law(state, context.validation_rows)
    prediction = np.sum(law.probabilities[:, None] * law.locations, axis=0)
    return float(np.sqrt(np.mean(np.square(prediction - context.validation_y))))


def _run_policy(context: RunContext, policy: str, config: dict[str, Any], oracle):
    state = _initial_state(context)
    available = list(range(len(context.candidate_rows)))
    initial_class_entropy = categorical_entropy(_structural_probabilities(context, state))
    curve = [_validation_rmse(context, state)]
    queries = []
    for round_index in range(int(config["acquisition_observation_budget"])):
        rows = tuple(context.candidate_rows[index] for index in available)
        scores, certified = _policy_scores(context, state, rows, policy, round_index, config)
        best_score = float(np.max(scores))
        tied = [index for index, score in enumerate(scores) if math.isclose(float(score), best_score, abs_tol=1e-15, rel_tol=0.0)]
        local = min(tied, key=lambda index: int(context.candidate_pool_indices[available[index]]))
        selected = available.pop(local)
        _, raw_target, returned = oracle.acquire_indices(
            np.asarray([context.candidate_pool_indices[selected]], dtype=int)
        )
        if int(returned[0]) != int(context.candidate_pool_indices[selected]):
            raise RuntimeError("candidate oracle changed the selected row identity")
        target = float(context.standardizer.transform_y(raw_target)[0])
        state = context.engine.update(
            state,
            context.candidate_rows[selected],
            target,
        )
        curve.append(_validation_rmse(context, state))
        queries.append({
            "dataset_id": context.dataset_id,
            "seed": context.seed,
            "policy": policy,
            "round": round_index + 1,
            "pool_index": int(context.candidate_pool_indices[selected]),
            "score": best_score,
            "ranking_certified": certified,
        })
    final_entropy = categorical_entropy(_structural_probabilities(context, state))
    values = np.asarray(curve)
    naulc = float(np.trapezoid(values) / ((len(values) - 1) * values[0]))
    return {
        "dataset_id": context.dataset_id,
        "seed": context.seed,
        "policy": policy,
        "status": "completed",
        "validation_rmse_initial": curve[0],
        "validation_rmse_final": curve[-1],
        "predictive_naulc": naulc,
        "frozen_structural_class_entropy_initial": initial_class_entropy,
        "frozen_structural_class_entropy_final": final_entropy,
        "frozen_structural_class_gain": initial_class_entropy - final_entropy,
        "ranking_certified_rate": float(np.mean([item["ranking_certified"] for item in queries])),
        "heldout_opened": False,
    }, queries


def _open_candidate_oracle(frame, config):
    prepared = prepare_real_selection(frame, split_seed=int(config["split_seed"]))
    return prepare_real_pool_oracle(
        frame, prepared, split_seed=int(config["split_seed"])
    )


def _family(dataset_id: str) -> str:
    return "uci_gas_turbine" if dataset_id.startswith("uci_gas_turbine_") else dataset_id


def _paired_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    radius = float(student_t.ppf(0.975, len(array) - 1) * np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean, mean - radius, mean + radius


def _family_paired_deltas(indexed, family: str, baseline: str, metric: str):
    seeds = sorted({seed for dataset, seed, policy in indexed if policy == PCPI_POLICY and _family(dataset) == family})
    values = []
    for seed in seeds:
        datasets = sorted({dataset for dataset, observed_seed, policy in indexed if observed_seed == seed and policy == PCPI_POLICY and _family(dataset) == family})
        values.append(float(np.mean([
            indexed[(dataset, seed, PCPI_POLICY)][metric]
            - indexed[(dataset, seed, baseline)][metric]
            for dataset in datasets
        ])))
    return values


def _assessment(runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    indexed = {(row["dataset_id"], row["seed"], row["policy"]): row for row in runs}
    effects = []
    for family in ("uci_ccpp", "uci_gas_turbine"):
        for baseline in ("random", "uncertainty", "qbc"):
            class_delta = _family_paired_deltas(
                indexed, family, baseline, "frozen_structural_class_gain"
            )
            risk_delta = _family_paired_deltas(
                indexed, family, baseline, "predictive_naulc"
            )
            effects.append({
                "family": family,
                "baseline": baseline,
                "class_gain_delta_mean_ci": _paired_interval(class_delta),
                "predictive_naulc_delta_mean_ci": _paired_interval(risk_delta),
                "negative_transfer_rate": float(np.mean(np.asarray(class_delta) < 0.0)),
            })
    random_effects = [item for item in effects if item["baseline"] == "random"]
    limit = float(config["assessment_rules"]["negative_transfer_rate_max"])
    strong = all(item["class_gain_delta_mean_ci"][1] > 0.0 and item["negative_transfer_rate"] <= limit for item in random_effects)
    strong = strong and all(item["predictive_naulc_delta_mean_ci"][0] <= 0.0 for item in effects)
    return {"status": "REAL_ADVANTAGE_DEMONSTRATED" if strong else "REAL_ADVANTAGE_NOT_DEMONSTRATED", "strong_evidence": strong, "paired_effects": effects}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _publish(output: Path, payload: dict[str, Any], calibration, runs, queries) -> None:
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "calibration_runs.csv", calibration)
    _write_csv(output / "acquisition_runs.csv", runs)
    _write_csv(output / "queries.csv", queries)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        path.name: _file_hash(path)
        for path in sorted(output.iterdir())
        if path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/p3g_1_structurewise_discrepancy_calibration_gate.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = _load_config(config_path)
    identity = _git_identity(root)
    dependency_snapshot = runtime_dependency_snapshot()
    started = time.perf_counter()
    contexts = []
    frames = {}
    dataset_hashes = {}
    for dataset_id in config["datasets"]:
        frame = load_registered_real_dataset(dataset_id, args.data_root, verify_hashes=True)
        frames[dataset_id] = frame
        dataset_hashes[dataset_id] = list(frame.source_hashes)
        for seed in config["seeds"]:
            contexts.append(_make_context(frame, config, int(seed)))
    calibration = []
    for index, context in enumerate(contexts, start=1):
        row = _calibration_run(context, config)
        calibration.append(row)
        print(
            f"[calibration {index:02d}/24] {context.dataset_id} "
            f"seed={context.seed} rejected={row['pit_rejected']}",
            flush=True,
        )
    eligible = len(calibration) == 24 and not any(row["pit_rejected"] for row in calibration)
    runs, queries = [], []
    if eligible:
        oracles = {
            dataset_id: _open_candidate_oracle(frame, config)
            for dataset_id, frame in frames.items()
        }
        for context in contexts:
            for policy in POLICIES:
                run, selected = _run_policy(
                    context, policy, config, oracles[context.dataset_id]
                )
                runs.append(run)
                queries.extend(selected)
                print(
                    f"[acquisition {len(runs):02d}/96] {context.dataset_id} "
                    f"seed={context.seed} policy={policy} status={run['status']}",
                    flush=True,
                )
    assessment = _assessment(runs, config) if eligible else {"status": "CALIBRATION_NO_GO", "strong_evidence": False, "paired_effects": []}
    payload = {
        "schema": "pcpi-p3g1-structurewise-discrepancy-gate-result-v1",
        "status": assessment["status"],
        "source_commit": identity["commit"],
        "source_tree": identity["tree"],
        "config_sha256": _file_hash(config_path),
        "production_code_hash": production_code_hash(root),
        "runtime_dependency_hash": runtime_dependency_hash(dependency_snapshot),
        "runtime_dependency_snapshot": dependency_snapshot,
        "dataset_source_hashes": dataset_hashes,
        "calibration_run_count": len(calibration),
        "calibration_rejection_count": sum(row["pit_rejected"] for row in calibration),
        "global_predictive_calibration_eligible": eligible,
        "acquisition_executed": eligible,
        "acquisition_run_count": len(runs),
        "heldout_opened": False,
        "simulated_experiment": False,
        "formal_development_efficacy_evidence": bool(assessment["strong_evidence"]),
        "formal_efficacy_evidence": False,
        "assessment": assessment,
        "wall_time_seconds": time.perf_counter() - started,
        "failure_policy": "fail-closed-record-all-no-seed-replacement-no-retry",
    }
    _publish(args.output_dir.resolve(), payload, calibration, runs, queries)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
