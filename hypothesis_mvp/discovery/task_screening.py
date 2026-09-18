"""Response-free task admission before provider, engine, or pool-response use."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis_mvp.data.system_protocol import (
    load_registered_system_data, validate_data_registration,
)
from hypothesis_mvp.pcpi.discovery_transaction import _publish
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .initializer import generic_deterministic_candidates
from .pcpi_adapter import structural_terms
from .system_run import audit_frozen_hypothesis_bank


SCHEMA = "scientific-task-response-free-screening-v1"


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def validate_task_screening_registration(config):
    required = {"schema", "data", "prior", "measurement_budget",
                "coefficient_policy", "exact_eig_epsabs",
                "user_execution_authorized"}
    if set(config) != required or config["schema"] != SCHEMA:
        raise ValueError("invalid task-screening registration")
    if (not config["data"] or len({row["dataset"] for row in config["data"]})
            != len(config["data"])):
        raise ValueError("task screening requires distinct registered datasets")
    for row in config["data"]:
        validate_data_registration(row)
        if row["counts"]["acquisition_pool"] < config["measurement_budget"]:
            raise ValueError("screening action domain is smaller than measurement budget")
    if (type(config["measurement_budget"]) is not int
            or config["measurement_budget"] < 1
            or config["coefficient_policy"]
            != "discard-fitted-coefficients-refit-closed-basis"
            or config["exact_eig_epsabs"] != 1e-10
            or type(config["user_execution_authorized"]) is not bool):
        raise ValueError("invalid task-screening controls")
    NormalInverseGammaPrior(**config["prior"])
    return config


def screen_loaded_task(data, registration, config):
    development = data.selection.development
    candidates = generic_deterministic_candidates(development.X, development.y)
    compatible, rejected = [], []
    for row in candidates:
        try:
            structural_terms(row["expression"], development.X.shape[1])
        except Exception as error:
            rejected.append({"source": row["source"],
                             "error_type": type(error).__name__,
                             "error": str(error)})
            continue
        compatible.append({**row, "origin": "deterministic-screening"})
    if len({tuple(structural_terms(row["expression"], development.X.shape[1]))
            for row in compatible}) < 2:
        raise ValueError("screening grammar retained fewer than two structural supports")
    audit = audit_frozen_hypothesis_bank(
        compatible, data.initial, data.pool.X_pool,
        n_features=data.initial.X.shape[1],
        prior=NormalInverseGammaPrior(**config["prior"]),
        exploration_identity=_digest({"registration": registration,
                                      "candidates": compatible}),
        coefficient_policy=config["coefficient_policy"],
        measurement_budget=config["measurement_budget"],
        exact_eig_epsabs=config["exact_eig_epsabs"],
    )
    return {"dataset": registration["dataset"], "passed": audit["passed"],
            "candidate_count": len(compatible), "adapter_rejections": rejected,
            "audit": audit, "provider_calls": 0, "engine_jobs": 0,
            "candidate_response_accessed": False, "heldout_opened": False}


def run_task_screening(root, config):
    validate_task_screening_registration(config)
    if config["user_execution_authorized"] is not True:
        raise PermissionError("task screening requires registered user execution")
    root = Path(root)
    if root.exists():
        raise ValueError("task-screening output must be a new path")
    root.mkdir(parents=True)
    rows = []
    try:
        for registration in config["data"]:
            print(f"response-free task screening: {registration['dataset']}", flush=True)
            data = load_registered_system_data(registration)
            row = screen_loaded_task(data, registration, config)
            rows.append(row)
            _publish(root / f"{registration['dataset']}.json", row)
        result = {"schema": "scientific-task-screening-result-v1", "rows": rows,
                  "admitted_tasks": [row["dataset"] for row in rows if row["passed"]],
                  "completed": True, "provider_calls": 0, "engine_jobs": 0,
                  "candidate_response_accessed": False, "heldout_opened": False,
                  "formal_experiment_authorized": False}
        _publish(root / "TASK_SCREENING.json", result)
        return result
    except BaseException as error:
        _publish(root / "TERMINAL_FAILURE.json", {
            "error_type": type(error).__name__, "completed_tasks": len(rows),
            "candidate_response_accessed": False, "heldout_opened": False})
        raise

