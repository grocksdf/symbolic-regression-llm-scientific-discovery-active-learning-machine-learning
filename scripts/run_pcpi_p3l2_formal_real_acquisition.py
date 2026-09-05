"""Run the frozen P3L.2 information-risk real-development protocol."""

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
    P3L_INFORMATION_RISK_METHOD,
    P3L_INFORMATION_RISK_RANK_CERTIFICATE,
    P3L_INFORMATION_RISK_TAIL_PROBABILITY,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run
from scripts.run_pcpi_p3k8_formal_real_acquisition import (
    PYTHON_EXECUTABLE_HASH,
    RUNTIME_BINARY_IDENTITY,
    RUNTIME_HASH,
    RUNTIME_IDENTITY_METHOD,
)


CONFIG_SHA256 = (
    "be210646f1e1a4aee93990c0e0954b4944c4f63847ddf149fba49ddd816719a6"
)
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)


def validate_p3l2_config(path: Path, root: Path) -> dict[str, object]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    if not resolved.is_file() or (resolved != project and project not in resolved.parents):
        raise ValueError("P3L.2 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3L.2 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": "pcpi-p3l2-formal-real-acquisition-config-v1",
        "stage": "P3L.2",
        "failed_predecessor": "P3K.8/real-advantage-not-demonstrated",
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
        "eig_rank_certificate_method": P3L_INFORMATION_RISK_RANK_CERTIFICATE,
        "pcpi_robust_utility": (
            "p3l-maximin-lower-tail-cvar-of-frozen-class-entropy-reduction"
        ),
        "pcpi_information_risk_method": P3L_INFORMATION_RISK_METHOD,
        "pcpi_information_risk_tail_probability": (
            P3L_INFORMATION_RISK_TAIL_PROBABILITY
        ),
        "pcpi_information_risk_tail_probability_source": (
            "p3k8-preregistered-negative-transfer-rate-maximum"
        ),
        "pcpi_mean_information_role": "audit-only-not-selection",
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
        raise ValueError("P3L.2 frozen formal contract changed")
    if config["assessment_rules"]["negative_transfer_rate_max"] != (
        config["pcpi_information_risk_tail_probability"]
    ):
        raise ValueError("P3L.2 risk tail is not the preregistered assessment tail")
    return config


CLAIM_BOUNDARY = (
    "P3L.2 is one failure-informed, held-out-closed real-development comparison. "
    "P3K.8 completed all registered real runs but did not demonstrate its primary "
    "class-entropy advantage: mean EIG did not control the registered 25% "
    "negative-transfer boundary. P3L.2 preserves datasets, seeds, budgets, "
    "baselines, posterior family, shared semiparametric response law, representative "
    "projection, runtime identity, and assessment. Before any candidate response is "
    "opened, it ranks by the minimum over all frozen likelihood powers of the 25% "
    "lower-tail CVaR of frozen-class entropy reduction under that same normalized "
    "joint law. The 25% tail is copied from the preregistered P3K.8 assessment, not "
    "estimated from its outcomes. Mean mutual information is audit-only. This is "
    "development evidence, not held-out confirmation or a guarantee against real "
    "distribution misspecification."
)


P3L2_PROTOCOL = RealAcquisitionProtocol(
    stage="P3L.2",
    schema="pcpi-p3l2-formal-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_information_risk_acquisition",
    hypothesis_id="pcpi-p3l2-real-information-risk-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3k8-real-advantage-not-demonstrated",
        "pcpi-p3l1-information-risk-correctness",
    ),
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    required_runtime_binary_identity=RUNTIME_BINARY_IDENTITY,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
    p3j_class_conditional_lifecycle=True,
    class_conditional_contract_prefix="p3k",
    class_conditional_representative_method=P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    class_conditional_singleton_rank_certificate=P3K_SINGLETON_RANK_CERTIFICATE,
    config_validator=validate_p3l2_config,
)


def main() -> int:
    return run(build_parser(P3L2_PROTOCOL, description=__doc__).parse_args(), P3L2_PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
