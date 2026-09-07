"""Run the frozen P3M.5 action-conditional real-development protocol."""

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
    P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    P3K_SINGLETON_RANK_CERTIFICATE,
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_JOINT_METHOD,
    P3M_ACTION_CONDITIONAL_POSTERIOR_UPDATE_METHOD,
    P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD,
    P3M_BANDWIDTH_RULE,
    P3M_BANDWIDTH_SCHEDULE,
    P3M_CHECKPOINT_PUBLICATION,
    P3M_CHECKPOINT_SCHEMA,
    P3M_CONTEXT_TRANSFORM,
    P3M_INFORMATION_RISK_RANK_CERTIFICATE,
    P3M_MEASURED_RUN_PROTOCOL,
    P3M_OPERATIONAL_LIFECYCLE,
    P3M_OUTER_RUNNER_COMPOSITION,
    P3M_POLICY_DISPATCH,
    P3M_REPORTING_ORDER,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run
from scripts.run_pcpi_p3k8_formal_real_acquisition import (
    PYTHON_EXECUTABLE_HASH,
    RUNTIME_BINARY_IDENTITY,
    RUNTIME_HASH,
    RUNTIME_IDENTITY_METHOD,
)


CONFIG_SHA256 = "310d7a616fa02b2c0c148c218a7235c06ebd74040e1da4eadc1425b1402fa95f"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)


def validate_p3m5_config(path: Path, root: Path) -> dict[str, object]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    if not resolved.is_file() or (resolved != project and project not in resolved.parents):
        raise ValueError("P3M.5 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.5 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": "pcpi-p3m5-formal-real-acquisition-config-v1",
        "stage": "P3M.5",
        "failed_predecessor": "P3L.2/real-advantage-not-demonstrated",
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
        "eig_rank_certificate_method": P3M_INFORMATION_RISK_RANK_CERTIFICATE,
        "pcpi_robust_utility": (
            "p3m-maximin-action-conditional-lower-tail-cvar-of-frozen-class-"
            "entropy-reduction"
        ),
        "pcpi_information_risk_method": P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
        "pcpi_information_risk_tail_probability": 0.25,
        "pcpi_information_risk_tail_probability_source": (
            "p3k8-preregistered-negative-transfer-rate-maximum"
        ),
        "pcpi_mean_information_role": "audit-only-not-selection",
        "representative_discrepancy": P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
        "p3m_operational_lifecycle": P3M_OPERATIONAL_LIFECYCLE,
        "p3m_residual_state_method": P3M_ACTION_CONDITIONAL_RESIDUAL_METHOD,
        "p3m_context_transform": P3M_CONTEXT_TRANSFORM,
        "p3m_bandwidth_rule": P3M_BANDWIDTH_RULE,
        "p3m_bandwidth_schedule": P3M_BANDWIDTH_SCHEDULE,
        "p3m_class_posterior_update": P3M_ACTION_CONDITIONAL_POSTERIOR_UPDATE_METHOD,
        "p3m_joint_information_method": P3M_ACTION_CONDITIONAL_JOINT_METHOD,
        "p3m_measured_run_protocol": P3M_MEASURED_RUN_PROTOCOL,
        "p3m_policy_dispatch": P3M_POLICY_DISPATCH,
        "p3m_reporting_order": P3M_REPORTING_ORDER,
        "p3m_outer_runner_composition": P3M_OUTER_RUNNER_COMPOSITION,
        "p3m_singleton_rank_certificate": P3K_SINGLETON_RANK_CERTIFICATE,
        "p3m_checkpoint_schema": P3M_CHECKPOINT_SCHEMA,
        "p3m_checkpoint_publication": P3M_CHECKPOINT_PUBLICATION,
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
        raise ValueError("P3M.5 frozen formal contract changed")
    if config["assessment_rules"]["negative_transfer_rate_max"] != 0.25:
        raise ValueError("P3M.5 must preserve the registered assessment")
    return config


CLAIM_BOUNDARY = (
    "P3M.5 is one failure-informed, held-out-closed real-development comparison. "
    "It preserves the P3L.2 datasets, seeds, budgets, baselines, posterior family, "
    "representative projection, risk tail, runtime identity, and assessment. Its "
    "only estimand change is a strict-prefix covariate-conditional shared-innovation "
    "law: each candidate is scored under its own normalized predictive response law, "
    "while that law remains shared across frozen classes for identifiability. Context "
    "construction is response-free and every revealed response advances all four "
    "likelihood-power states exactly once. This is development evidence, not held-out "
    "confirmation or a guarantee against real distribution misspecification."
)


P3M5_PROTOCOL = RealAcquisitionProtocol(
    stage="P3M.5",
    schema="pcpi-p3m5-formal-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_action_conditional_risk_acquisition",
    hypothesis_id="pcpi-p3m5-real-action-conditional-risk-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3l2-real-advantage-not-demonstrated",
        "pcpi-p3m1-through-p3m4-action-conditional-correctness",
    ),
    required_runtime_dependency_hash=RUNTIME_HASH,
    required_python_executable_hash=PYTHON_EXECUTABLE_HASH,
    required_runtime_binary_identity=RUNTIME_BINARY_IDENTITY,
    shared_initial_frozen_target=True,
    fail_fast=True,
    operational_execution_authorized=True,
    p3j_class_conditional_lifecycle=True,
    class_conditional_contract_prefix="p3m",
    class_conditional_representative_method=P3K_PROJECTED_REPRESENTATIVE_MMD_METHOD,
    class_conditional_singleton_rank_certificate=P3K_SINGLETON_RANK_CERTIFICATE,
    config_validator=validate_p3m5_config,
)


def main() -> int:
    return run(build_parser(P3M5_PROTOCOL, description=__doc__).parse_args(), P3M5_PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
