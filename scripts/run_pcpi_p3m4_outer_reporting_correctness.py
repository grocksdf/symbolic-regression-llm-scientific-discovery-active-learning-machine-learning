"""Run the no-data P3M.4 outer-policy and reporting Gate."""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.pcpi import (
    P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
    P3M_ACTION_CONDITIONAL_JOINT_METHOD,
    P3M_OUTER_RUNNER_COMPOSITION,
    P3M_POLICY_DISPATCH,
    P3M_POLICY_FAILURE_SCHEMA,
    P3M_REPORTING_ORDER,
)
from hypothesis_mvp.pcpi import (
    p3j_outer_runner as outer,
    p3j_policy_integration as dispatch,
    p3j_reporting as reporting,
)


CONFIG = PROJECT_ROOT / "configs" / "p3m_4_outer_reporting_correctness.json"
CONFIG_SHA256 = "c665fbdd8938a0b1f725a56aa331c47c23ec1935daf2ca63c0302e0a05e5d648"


def _load_config() -> dict[str, object]:
    raw = CONFIG.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3M.4 config hash changed")
    config = json.loads(raw.decode("utf-8"))
    expected = {
        "stage": "P3M.4",
        "policy_dispatch": P3M_POLICY_DISPATCH,
        "failure_schema": P3M_POLICY_FAILURE_SCHEMA,
        "reporting_order": P3M_REPORTING_ORDER,
        "outer_composition": P3M_OUTER_RUNNER_COMPOSITION,
        "decision_audit_method": P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD,
        "expected_joint_method": P3M_ACTION_CONDITIONAL_JOINT_METHOD,
        "real_data_access_authorized": False,
        "operational_execution_authorized": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("P3M.4 contract changed")
    return config


def _evaluate(_: dict[str, object]) -> dict[str, object]:
    route = inspect.getsource(dispatch.dispatch_p3j_matched_policy)
    failure = inspect.getsource(dispatch.publish_p3j_policy_failure_snapshot)
    build = inspect.getsource(reporting.build_p3j_policy_artifacts)
    audit = inspect.getsource(reporting._p3j_decision_valid)
    summarize = inspect.getsource(reporting.summarize_p3j_policy_artifacts)
    runner = inspect.getsource(outer.run_p3j_outer_policy)
    decisions = {
        "only_pcpi_uses_transactional_state": (
            "if policy == pcpi_policy" in route and "legacy_runner()" in route
        ),
        "failure_snapshot_forbids_retry_and_seed_replacement": (
            "retry_in_same_run_forbidden" in failure
            and "seed_replacement_forbidden" in failure
        ),
        "p3m_reporting_identity_is_separate": (
            "P3M_REPORTING_ORDER" in build
            and "P3M_ACTION_CONDITIONAL_JOINT_METHOD" in build
        ),
        "audit_requires_p3m_cvar_and_joint": (
            "P3M_ACTION_CONDITIONAL_INFORMATION_RISK_METHOD" in audit
            and "expected_joint_method" in audit
        ),
        "reporting_retains_score_gain_correlation": (
            "score_realized_gain_spearman" in summarize
        ),
        "outer_order_is_transaction_then_reporting_then_summary": (
            runner.index("run_p3j_measured_pool_acquisition")
            < runner.index("build_p3j_policy_artifacts")
            < runner.index("summarize_p3j_policy_artifacts")
        ),
        "outer_failure_uses_p3m_schema": (
            "P3M_POLICY_FAILURE_SCHEMA" in runner
            and "P3M_OUTER_RUNNER_COMPOSITION" in runner
        ),
    }
    if not all(decisions.values()):
        failed = [name for name, passed in decisions.items() if not passed]
        raise AssertionError(f"P3M.4 correctness Gate failed: {failed}")
    return {
        "schema": "pcpi-p3m4-outer-reporting-correctness-gate-v1",
        "stage": "P3M.4",
        "status": "passed-correctness-formal-freeze-blocked",
        "role": "outer-policy-failure-reporting-audit-gate",
        "decisions": decisions,
        "formal_experiment": False,
        "real_data_access": False,
        "heldout_access": False,
        "simulated_experiment": False,
        "operational_execution_authorized": False,
    }


def main() -> int:
    print(json.dumps(_evaluate(_load_config()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
