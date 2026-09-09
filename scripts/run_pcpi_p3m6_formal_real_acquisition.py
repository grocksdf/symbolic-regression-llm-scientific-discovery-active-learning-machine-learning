"""Run the frozen P3M.6 global--local partial-pooling protocol."""

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
    P3M_CHECKPOINT_PUBLICATION,
    P3M_CHECKPOINT_SCHEMA,
    P3M_CONTEXT_TRANSFORM,
    P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
    P3M_GLOBAL_LOCAL_POOLING_KAPPA,
    P3M_GLOBAL_LOCAL_POOLING_RULE,
    P3M_INFORMATION_RISK_RANK_CERTIFICATE,
    P3M_MEASURED_RUN_PROTOCOL,
    P3M_OPERATIONAL_LIFECYCLE,
    P3M_OUTER_RUNNER_COMPOSITION,
    P3M_POLICY_DISPATCH,
    P3M_REPORTING_ORDER,
)
from scripts.run_pcpi_p3b_real import RealAcquisitionProtocol, build_parser, run
from scripts.run_pcpi_p3m5_formal_real_acquisition import (
    RUNTIME_BINARY_IDENTITY,
    RUNTIME_HASH,
    RUNTIME_IDENTITY_METHOD,
    PYTHON_EXECUTABLE_HASH,
    validate_p3m5_config,
)


CONFIG_SHA256 = "1ee7c12a9bfbdd85f20973ef4f50f805148bfc26e25f5916e1ba971268f07741"
POLICIES = ("random", "uncertainty", "qbc", DECISION_TARGETED_POLICY)
CONFIG_NAME = "p3m_6_global_local_partial_pooling_real_acquisition.json"


def validate_p3m6_config(path: Path, root: Path) -> dict[str, object]:
    resolved, project = Path(path).resolve(), Path(root).resolve()
    if not resolved.is_file() or (resolved != project and project not in resolved.parents):
        raise ValueError("P3M.6 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.6 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    baseline = validate_p3m5_config(
        project / "configs" / "p3m_5_action_conditional_real_acquisition.json",
        project,
    )
    expected = dict(baseline)
    expected.update({
        "schema": "pcpi-p3m6-global-local-partial-pooling-real-acquisition-config-v1",
        "stage": "P3M.6",
        "failed_predecessor": "P3M.5/real-advantage-not-demonstrated",
        "p3m_residual_state_method": P3M_GLOBAL_LOCAL_PARTIAL_POOLED_RESIDUAL_METHOD,
        "p3m_residual_pooling_kappa": P3M_GLOBAL_LOCAL_POOLING_KAPPA,
        "p3m_residual_pooling_rule": P3M_GLOBAL_LOCAL_POOLING_RULE,
        "operational_execution_authorized": True,
        "formal_dataset_runner_authorized": True,
    })
    expected["datasets"] = list(P2A_REAL_DATASETS)
    expected["policies"] = list(POLICIES)
    expected["seeds"] = list(range(2026080701, 2026080709))
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"P3M.6 frozen formal contract changed: {key}")
    if set(config) != set(expected):
        raise ValueError("P3M.6 config contains an unregistered field")
    if config["assessment_rules"]["negative_transfer_rate_max"] != 0.25:
        raise ValueError("P3M.6 must preserve the registered assessment")
    return config


CLAIM_BOUNDARY = (
    "P3M.6 is a held-out-closed real-development comparison following the "
    "completed P3M.5 negative result. It preserves the P3M.5 datasets, seeds, "
    "budgets, baselines, posterior family, representative projection, risk tail, "
    "runtime identity, and assessment. Its sole estimand change is a strict-prefix "
    "global--local residual law: each candidate retains its normalized RBF law but "
    "shrinks toward the same global PIT law with lambda=n_eff/(n_eff+8), where the "
    "fixed kappa=8 is a response-free registered prior effective sample size. "
    "No validation, held-out, oracle, or outcome-dependent tuning enters the law. "
    "This is development evidence, not confirmation or a guarantee against real "
    "distribution misspecification."
)


P3M6_PROTOCOL = RealAcquisitionProtocol(
    stage="P3M.6",
    schema="pcpi-p3m6-global-local-partial-pooling-real-acquisition-config-v1",
    experiment="real_measurement_matched_budget_global_local_partial_pooling_acquisition",
    hypothesis_id="pcpi-p3m6-real-global-local-partial-pooling-acquisition",
    pcpi_policy=DECISION_TARGETED_POLICY,
    policies=POLICIES,
    claim_boundary=CLAIM_BOUNDARY,
    parent_lineage=(
        "pcpi-p3m5-real-advantage-not-demonstrated",
        "pcpi-p3m6-global-local-partial-pooling-correctness",
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
    config_validator=validate_p3m6_config,
)


def main() -> int:
    return run(build_parser(P3M6_PROTOCOL, description=__doc__).parse_args(), P3M6_PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
