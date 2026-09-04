"""Run the frozen P3K.8 complete-runtime-identity real protocol."""

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
    P3K_SINGLETON_RANK_CERTIFICATE,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run


RUNTIME_HASH = (
    "6b8c2611d77caea375d0d90d2e84627c725f0df845194bff14e25d44b4492ad9"
)
PYTHON_EXECUTABLE_HASH = (
    "560b9ef7d856608ab8da02ded2dc8a1951ad1f424c382c0ec6a698874165a18e"
)
RUNTIME_IDENTITY_METHOD = (
    "dependency-snapshot-plus-launcher-base-python-and-abi-dll-hashes-v1"
)
RUNTIME_BINARY_IDENTITY = {
    "base_executable": {
        "length": 91648,
        "sha256": "d8e3f0adf246db00358c0c4ed349cf714898178f9558fb0e944f79f5c07f8eaa",
    },
    "python_dll": {
        "length": 6969856,
        "sha256": "64a1dad031e97f13b1a0bac26c689d8e14a18d7dd1eab06e17f70e22373f4eec",
    },
    "stable_abi_dll": {
        "length": 56320,
        "sha256": "2d2330ce33d1443c67b1804bbf1a561653499dd4dcf8918048747cc17d1a63c4",
    },
    "venv_launcher": {
        "length": 262144,
        "sha256": PYTHON_EXECUTABLE_HASH,
    },
    "pyvenv_config": {
        "length": 306,
        "sha256": "215ca7981b0a4d50d5eee7a23d5157c05d4ea867e8182c04f3cfcc26bdaf5106",
    },
}
CONFIG_SHA256 = (
    "ae51f789cff79523f4c00ed364d08afea74878ded09d847419bb4de50479bdd9"
)
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)


def validate_p3k8_config(path: Path, root: Path) -> dict[str, object]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    if not resolved.is_file() or (
        resolved != project and project not in resolved.parents
    ):
        raise ValueError("P3K.8 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3K.8 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": "pcpi-p3k8-formal-real-acquisition-config-v1",
        "stage": "P3K.8",
        "failed_predecessor": "P3K.7/pre-data-mutable-base-runtime-drift",
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
        "representative_safe_set_rule": (
            "nonincrease-if-feasible-else-minimum-attainable-mmd-increase"
        ),
        "representative_empty_safe_set_action": (
            "minimum-violation-projection-no-utility-switch"
        ),
        "p3k_operational_lifecycle": P3K_OPERATIONAL_LIFECYCLE,
        "p3k_residual_state_method": P3K_SHARED_INNOVATION_RESIDUAL_METHOD,
        "p3k_class_posterior_update": P3K_PREQUENTIAL_POSTERIOR_UPDATE_METHOD,
        "p3k_joint_information_method": P3K_SHARED_INNOVATION_JOINT_METHOD,
        "p3k_measured_run_protocol": P3K_MEASURED_RUN_PROTOCOL,
        "p3k_policy_dispatch": P3K_POLICY_DISPATCH,
        "p3k_reporting_order": P3K_REPORTING_ORDER,
        "p3k_outer_runner_composition": P3K_OUTER_RUNNER_COMPOSITION,
        "p3k_singleton_rank_certificate": P3K_SINGLETON_RANK_CERTIFICATE,
        "runtime_dependency_hash": RUNTIME_HASH,
        "runtime_identity_method": RUNTIME_IDENTITY_METHOD,
        "runtime_binary_identity": RUNTIME_BINARY_IDENTITY,
        "runtime_storage": "workspace-local-copy-not-managed-runtime-symlink",
        "split_seed": SPLIT_SEED,
        "heldout_state": "closed",
        "failure_policy": "fail_fast_record_terminal_no_seed_replacement",
        "operational_execution_authorized": True,
        "formal_dataset_runner_authorized": True,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise ValueError("P3K.8 frozen formal contract changed")
    return config


CLAIM_BOUNDARY = (
    "P3K.8 is one failure-informed, held-out-closed real-development comparison. "
    "It preserves every P3K.7 statistical and experimental choice. P3K.7 reached "
    "no data because its unchanged virtual-environment launcher resolved through a "
    "managed base interpreter that had changed from CPython 3.12.13 to 3.12.14. "
    "P3K.8 restores 3.12.13 in a workspace-local runtime and freezes the complete "
    "dependency snapshot, launcher, base executable, versioned Python DLL, stable-ABI "
    "DLL, and virtual-environment configuration before data-path validation. No "
    "dataset, seed, budget, baseline, model, utility, projection, ranking, assessment "
    "or held-out rule changes. This is development evidence, not confirmation."
)


P3K8_PROTOCOL = RealAcquisitionProtocol(
    stage="P3K.8",
    schema="pcpi-p3k8-formal-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_p3k_complete_runtime_identity",
    hypothesis_id="pcpi-p3k8-real-shared-innovation-complete-runtime-identity",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3k7-pre-data-mutable-base-runtime-drift",
        "pcpi-p3k8-complete-runtime-identity-correctness",
    ),
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    required_runtime_binary_identity=RUNTIME_BINARY_IDENTITY,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
    p3j_class_conditional_lifecycle=True,
    class_conditional_contract_prefix="p3k",
    class_conditional_representative_method=(
        P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD
    ),
    class_conditional_singleton_rank_certificate=P3K_SINGLETON_RANK_CERTIFICATE,
    config_validator=validate_p3k8_config,
)


def main() -> int:
    return run(
        build_parser(P3K8_PROTOCOL, description=__doc__).parse_args(),
        P3K8_PROTOCOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
