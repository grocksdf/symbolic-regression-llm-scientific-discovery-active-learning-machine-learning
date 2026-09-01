"""Blocked P3J.14 shared outer-runner composition."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from hypothesis_mvp.pcpi import (
    P3H_OPERATIONAL_POWERS,
    P3J_CLASS_CONDITIONAL_JOINT_METHOD,
    P3J_CLASS_CONDITIONAL_RESIDUAL_METHOD,
    P3J_CLASS_POSTERIOR_UPDATE_METHOD,
    P3J_MEASURED_RUN_PROTOCOL,
    P3J_OPERATIONAL_LIFECYCLE,
    P3J_OUTER_RUNNER_COMPOSITION,
    P3J_POLICY_DISPATCH,
    P3J_REPORTING_ORDER,
    OperationalClassConditionalState,
    dispatch_p3j_matched_policy,
    run_p3j_outer_policy,
)
from scripts.run_pcpi_p3b_real import _run_policy


STAGE = "P3J.14"
SCHEMA = "pcpi-p3j14-shared-outer-runner-config-v1"
PCPI_POLICY = "pcpi_representative_safe_robust_class_eig"
CONFIG_SHA256 = "3ac1dd6492d7b9700735799e3d80578f3bfdffabbd26b08b75328d347a0de1d6"


def _load_config(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    if not resolved.is_file() or (resolved != root and root not in resolved.parents):
        raise ValueError("P3J.14 config must be inside the project root")
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("P3J.14 frozen config hash changed")
    config = json.loads(raw.decode("utf-8"))
    frozen = {
        "schema": SCHEMA,
        "stage": STAGE,
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
        "heldout_state": "closed",
        "failure_policy": "fail_fast_record_terminal_no_seed_replacement",
        "operational_execution_authorized": False,
        "formal_dataset_runner_authorized": False,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise ValueError("P3J.14 frozen outer-runner contract changed")
    if tuple(config.get("policies", ())) != (
        "random", "uncertainty", "qbc", PCPI_POLICY
    ):
        raise ValueError("P3J.14 matched policy set changed")
    return config


def _compose_policy_call(
    policy: str,
    p3j_state: OperationalClassConditionalState | None,
    p3j_kwargs: dict[str, Any],
    legacy_kwargs: dict[str, Any],
    *,
    p3j_call: Callable[..., object] = run_p3j_outer_policy,
    legacy_call: Callable[..., object] = _run_policy,
) -> object:
    return dispatch_p3j_matched_policy(
        policy,
        PCPI_POLICY,
        p3j_state,
        lambda: p3j_call(**p3j_kwargs),
        lambda: legacy_call(**legacy_kwargs),
    )


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    config = _load_config(Path(args.config), root)
    if (
        config["operational_execution_authorized"] is not False
        or config["formal_dataset_runner_authorized"] is not False
    ):
        raise AssertionError("P3J.14 authorization must remain closed")
    raise PermissionError(
        "P3J.14 is source-composition-only; registered real data access is blocked"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--heldout-state", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
