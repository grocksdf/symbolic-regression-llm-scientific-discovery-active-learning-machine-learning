"""Run the frozen P3K.5 projected-guard transactional real protocol."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.data import P2A_REAL_DATASETS, SPLIT_SEED
from hypothesis_mvp.pcpi import (
    DECISION_TARGETED_POLICY,
    P3H_OPERATIONAL_POWERS,
    P3K_MEASURED_RUN_PROTOCOL,
    P3K_OPERATIONAL_LIFECYCLE,
    P3K_OUTER_RUNNER_COMPOSITION,
    P3K_POLICY_DISPATCH,
    P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD,
    P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    P3K_REPORTING_ORDER,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
    P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = "6b8c2611d77caea375d0d90d2e84627c725f0df845194bff14e25d44b4492ad9"
PYTHON_EXECUTABLE_HASH = "560b9ef7d856608ab8da02ded2dc8a1951ad1f424c382c0ec6a698874165a18e"
CONFIG_SHA256 = "34ceace6510c6a7a56b8e9bcae0f012b9849cb74c11d0a16ba4a8a5b69cfffd8"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)


def validate_p3k5_config(path: Path, root: Path) -> dict[str, object]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    if not resolved.is_file() or (resolved != project and project not in resolved.parents):
        raise ValueError("P3K.5 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3K.5 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": "pcpi-p3k5-formal-real-acquisition-config-v1",
        "stage": "P3K.5",
        "failed_predecessor": "P3K.3/terminal-empty-representative-safe-set",
        "datasets": list(P2A_REAL_DATASETS),
        "policies": list(POLICIES),
        "seeds": list(range(2026080701, 2026080709)),
        "initial_observation_budget": 32,
        "initial_base_warmup_budget": 16,
        "initial_residual_training_budget": 16,
        "acquisition_observation_budget": 32,
        "candidate_pool_budget": 128,
        "validation_budget": 256,
        "likelihood_power_candidates": list(P3H_OPERATIONAL_POWERS),
        "representative_discrepancy": P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
        "representative_safe_set_rule": "nonincrease-if-feasible-else-minimum-attainable-mmd-increase",
        "representative_empty_safe_set_action": "minimum-violation-projection-no-utility-switch",
        "p3k_operational_lifecycle": P3K_OPERATIONAL_LIFECYCLE,
        "p3k_residual_state_method": P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
        "p3k_class_posterior_update": P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD,
        "p3k_joint_information_method": P3K_SHARED_INNOVATION_JOINT_METHOD,
        "p3k_measured_run_protocol": P3K_MEASURED_RUN_PROTOCOL,
        "p3k_policy_dispatch": P3K_POLICY_DISPATCH,
        "p3k_reporting_order": P3K_REPORTING_ORDER,
        "p3k_outer_runner_composition": P3K_OUTER_RUNNER_COMPOSITION,
        "runtime_dependency_hash": RUNTIME_HASH,
        "split_seed": SPLIT_SEED,
        "heldout_state": "closed",
        "failure_policy": "fail_fast_record_terminal_no_seed_replacement",
        "operational_execution_authorized": True,
        "formal_dataset_runner_authorized": True,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise ValueError("P3K.5 frozen formal contract changed")
    return config


CLAIM_BOUNDARY = (
    "P3K.5 is one failure-informed, held-out-closed real-development comparison. "
    "It preserves P3K shared-innovation inference and replaces only the infeasible "
    "sequential representative constraint by its covariate-only minimum-violation "
    "projection: nonincreasing candidates are unchanged when any exist; otherwise "
    "only minimum-MMD-increase candidates are eligible for the same P3K utility. "
    "No response, validation target, held-out value, dataset label, or empirical "
    "effect size enters this projection. All datasets, seeds, budgets, baselines, "
    "numerical schedules, runtime, source and configuration are frozen. This is "
    "development evidence, not independent confirmation or a universal claim."
)


P3K5_PROTOCOL = RealAcquisitionProtocol(
    stage="P3K.5",
    schema="pcpi-p3k5-formal-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3k_projected_guard_acquisition",
    hypothesis_id="pcpi-p3k5-real-shared-innovation-projected-guard-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3k3-terminal-empty-representative-safe-set",
        "pcpi-p3k4-minimum-violation-projection-correctness",
    ),
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
    p3j_class_conditional_lifecycle=True,
    class_conditional_contract_prefix="p3k",
    class_conditional_representative_method=P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    config_validator=validate_p3k5_config,
)


def main() -> int:
    return run(build_parser(P3K5_PROTOCOL, description=__doc__).parse_args(), P3K5_PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
