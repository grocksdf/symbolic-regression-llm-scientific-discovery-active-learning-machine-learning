"""Run the frozen P3J.15 class-conditional transactional real protocol."""

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
    P3J_CLASS_CONDITIONAL_JOINT_METHOD,
    P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD,
    P3J_CLASS_POSTERIOR_UPDATE_METHOD,
    P3J_MEASURED_RUN_PROTOCOL,
    P3J_OPERATIONAL_LIFECYCLE,
    P3J_OUTER_RUNNER_COMPOSITION,
    P3J_POLICY_DISPATCH,
    P3J_REPORTING_ORDER,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = "6b8c2611d77caea375d0d90d2e84627c725f0df845194bff14e25d44b4492ad9"
PYTHON_EXECUTABLE_HASH = (
    "560b9ef7d856608ab8da02ded2dc8a1951ad1f424c382c0ec6a698874165a18e"
)
CONFIG_SHA256 = "15111d61cbf653ca1a0f69af9deaf8b9afb3f49f4e8b97ade309c9b24ec856c3"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)


def validate_p3j15_config(path: Path, root: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    project = Path(root).resolve()
    if not resolved.is_file() or (resolved != project and project not in resolved.parents):
        raise ValueError("P3J.15 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3J.15 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": "pcpi-p3j15-formal-real-acquisition-config-v1",
        "stage": "P3J.15",
        "datasets": list(P2A_REAL_DATASETS),
        "policies": list(POLICIES),
        "seeds": list(range(2026080701, 2026080709)),
        "initial_observation_budget": 32,
        "initial_base_warmup_budget": 16,
        "initial_residual_training_budget": 16,
        "acquisition_observation_budget": 32,
        "candidate_pool_budget": 128,
        "validation_budget": 256,
        "eig_quadrature_min_evaluations": 32,
        "eig_quadrature_max_evaluations": 512,
        "eig_quadrature_growth_factor": 2,
        "eig_quadrature_error_safety_factor": 4.0,
        "eig_action_chunk_size": 16,
        "likelihood_power_candidates": list(P3H_OPERATIONAL_POWERS),
        "p3j_operational_lifecycle": P3J_OPERATIONAL_LIFECYCLE,
        "p3j_residual_state_method": P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD,
        "p3j_class_posterior_update": P3J_CLASS_POSTERIOR_UPDATE_METHOD,
        "p3j_joint_information_method": P3J_CLASS_CONDITIONAL_JOINT_METHOD,
        "p3j_measured_run_protocol": P3J_MEASURED_RUN_PROTOCOL,
        "p3j_policy_dispatch": P3J_POLICY_DISPATCH,
        "p3j_reporting_order": P3J_REPORTING_ORDER,
        "p3j_outer_runner_composition": P3J_OUTER_RUNNER_COMPOSITION,
        "runtime_dependency_hash": RUNTIME_HASH,
        "split_seed": SPLIT_SEED,
        "heldout_state": "closed",
        "failure_policy": "fail_fast_record_terminal_no_seed_replacement",
        "operational_execution_authorized": True,
        "formal_dataset_runner_authorized": True,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise ValueError("P3J.15 frozen formal contract changed")
    return config


CLAIM_BOUNDARY = (
    "P3J.15 is one failure-informed held-out-closed real-development comparison. "
    "PCPI alone uses the complete four-power class-conditional semiparametric "
    "state and a source/config/dataset/seed/query/state-bound resumable transaction. "
    "Each complete response-free decision is durably published before exactly one "
    "matching measured-pool response is opened; validation is evaluated only after "
    "the response ledger and cannot affect later selection. Random, uncertainty, "
    "and QBC retain the established matched shared runner. All registered datasets, "
    "eight seeds, initial roles, budgets, action domain, quadrature schedule, "
    "ambiguity models, tie breaks, runtime, source and configuration are frozen. "
    "Held-out remains sealed. The completed run is development evidence, not an "
    "independent confirmation, universal superiority claim, scientific law, or "
    "guarantee of paper acceptance. Negative or terminal results must be preserved."
)


P3J15_PROTOCOL = RealAcquisitionProtocol(
    stage="P3J.15",
    schema="pcpi-p3j15-formal-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3j_class_conditional_acquisition",
    hypothesis_id="pcpi-p3j15-real-class-conditional-transactional-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3j14-shared-outer-runner-gate",
        "pcpi-p3j15-formal-execution-freeze",
    ),
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
    p3j_class_conditional_lifecycle=True,
    config_validator=validate_p3j15_config,
)


def main() -> int:
    return run(
        build_parser(P3J15_PROTOCOL, description=__doc__).parse_args(),
        P3J15_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
