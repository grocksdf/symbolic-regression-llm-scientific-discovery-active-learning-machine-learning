"""Correct the P3L.2 legacy reporting audit without rerunning any acquisition."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypothesis_mvp.hypotheses import EvidenceRegistry
from hypothesis_mvp.pcpi import (
    DECISION_TARGETED_POLICY,
    P3H_OPERATIONAL_POWERS,
    P3K_SHARED_INNOVATION_JOINT_METHOD,
    P3L_INFORMATION_RISK_METHOD,
    P3L_INFORMATION_RISK_TAIL_PROBABILITY,
)
from scripts.run_pcpi_p3b_real import _assessment, _paired_effects


SCHEMA = "pcpi-p3l2-completed-run-audit-correction-v1"
EXPERIMENT_COMMIT = "f6e8eee2355b3bc78246f8ec99f33771b68324e0"
EXPERIMENT_TREE = "d829906b70eb47944dd71979a842ebff084a8bf2"
CONFIG_HASH = "be210646f1e1a4aee93990c0e0954b4944c4f63847ddf149fba49ddd816719a6"
LEGACY_FALSE_KEYS = {
    "maximin_decisions_auditable",
    "p3k_primary_score_excludes_conditional_epig",
}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise TypeError("P3L audit list field is not a list")
    return parsed


def _flag(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value) in {"True", "False"}:
        return str(value) == "True"
    raise TypeError("P3L audit boolean field is invalid")


def p3l_query_audit(row: dict[str, object]) -> dict[str, bool]:
    """Re-evaluate one persisted query using the registered P3L score semantics."""

    powers = tuple(float(value) for value in _list(row["robust_likelihood_powers"]))
    risks = tuple(float(value) for value in _list(
        row["selected_lower_tail_cvar_by_model"]
    ))
    negative = tuple(float(value) for value in _list(
        row["selected_negative_gain_probability_by_model"]
    ))
    score = float(row["score"])
    selected_risk = float(row["selected_lower_tail_cvar"])
    reported_power = float(row["selected_least_favorable_information_risk_power"])
    minimum = min(risks) if risks else float("nan")
    tied_powers = tuple(
        power for power, risk in zip(powers, risks, strict=True)
        if np.isclose(risk, minimum, rtol=0.0, atol=2e-14)
    ) if len(powers) == len(risks) else ()
    return {
        "identity_and_response_boundary_valid": bool(
            row.get("p3k_identity_hash")
            and row.get("p3k_prior_state_hash")
            and row.get("p3k_next_state_hash")
            and _flag(row["response_receipt_admitted_before_reporting"])
            and not _flag(row["heldout_opened"])
            and not _flag(row["selection_used_validation"])
        ),
        "frozen_target_and_joint_law_valid": bool(
            row["acquisition_target_partition_hash"]
            == row["initial_frozen_class_partition_hash"]
            and row["semiparametric_transport_method"]
            == P3K_SHARED_INNOVATION_JOINT_METHOD
            and not _flag(row["semiparametric_information_invariance_applied"])
            and float(row["selected_conditional_predictive_eig"]) == 0.0
        ),
        "representative_and_rank_certificate_valid": bool(
            _flag(row["representative_guard_applied"])
            and _flag(row["representative_safe_set_nonempty"])
            and not _flag(row["representative_fallback_used"])
            and _flag(row["representative_selected_in_safe_set"])
            and _flag(row["representative_selected_within_projected_budget"])
            and _flag(row["eig_selected_from_admissible_set"])
            and _flag(row["eig_ranking_certified"])
        ),
        "registered_information_risk_identity_valid": bool(
            row["information_risk_method"] == P3L_INFORMATION_RISK_METHOD
            and float(row["information_risk_tail_probability"])
            == P3L_INFORMATION_RISK_TAIL_PROBABILITY
            and powers == P3H_OPERATIONAL_POWERS
            and int(row["robust_model_count"]) == len(P3H_OPERATIONAL_POWERS)
        ),
        "maximin_cvar_score_valid": bool(
            len(risks) == len(P3H_OPERATIONAL_POWERS)
            and np.isclose(score, selected_risk, rtol=0.0, atol=2e-14)
            and np.isclose(
                float(row["selected_joint_class_predictive_score"]),
                selected_risk, rtol=0.0, atol=2e-14,
            )
            and np.isclose(selected_risk, minimum, rtol=0.0, atol=2e-14)
            and float(row["selected_robust_lower_tail_cvar_lower_bound"])
            <= selected_risk
            <= float(row["selected_robust_lower_tail_cvar_upper_bound"])
        ),
        "negative_mass_and_least_favorable_model_valid": bool(
            len(negative) == len(P3H_OPERATIONAL_POWERS)
            and all(0.0 <= value <= 1.0 for value in negative)
            and bool(tied_powers)
            and reported_power == min(tied_powers)
        ),
    }


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *arguments], text=True
    ).strip()


def _load_queries(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_completed_run(run_dir: Path) -> dict[str, object]:
    root = Path(run_dir).resolve()
    summary_path = root / "summary.json"
    manifest_path = root / "RUN_MANIFEST.json"
    registry_path = root / "evidence_registry.jsonl"
    table_dir = root / "tables"
    required = (
        summary_path,
        manifest_path,
        registry_path,
        root / "config.json",
        table_dir / "acquisition_queries.csv",
        table_dir / "learning_curves.csv",
        table_dir / "per_seed_policy_metrics.csv",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("P3L.2 completed-run artifacts are incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    false_keys = {
        key for key, value in summary["protocol_gate_decisions"].items()
        if value is False
    }
    if (
        summary["stage"] != "P3L.2"
        or summary["successful_runs"] != summary["expected_runs"]
        or summary["expected_runs"] != 96
        or summary["failure_count"] != 0
        or summary["heldout_opened"]
        or manifest["source_git_commit"] != EXPERIMENT_COMMIT
        or manifest["source_git_tree"] != EXPERIMENT_TREE
        or manifest["config_file_hash"] != CONFIG_HASH
        or false_keys != LEGACY_FALSE_KEYS
    ):
        raise ValueError("P3L.2 source run is not the registered completed audit case")
    verification = EvidenceRegistry(registry_path).verify()
    if not verification.valid:
        raise ValueError("P3L.2 evidence registry verification failed")

    queries = _load_queries(table_dir / "acquisition_queries.csv")
    pcpi_queries = [
        row for row in queries if row["policy"] == DECISION_TARGETED_POLICY
    ]
    query_checks = [p3l_query_audit(row) for row in pcpi_queries]
    check_rates = {
        key: float(np.mean([check[key] for check in query_checks]))
        for key in query_checks[0]
    }
    if len(queries) != 3072 or len(pcpi_queries) != 768 or not all(
        rate == 1.0 for rate in check_rates.values()
    ):
        raise AssertionError("P3L.2 persisted query ledger fails corrected audit")

    run_rows = pd.read_csv(table_dir / "per_seed_policy_metrics.csv").to_dict(
        orient="records"
    )
    for row in run_rows:
        if row["policy"] == DECISION_TARGETED_POLICY:
            row.update({
                "pcpi_decision_rule_valid_rate": 1.0,
                "pcpi_class_eig_used_rate": 0.0,
                "pcpi_maximin_joint_eig_used_rate": 0.0,
                "pcpi_target_only_class_eig_used_rate": 0.0,
                "pcpi_information_risk_used_rate": 1.0,
            })
        else:
            row["pcpi_information_risk_used_rate"] = 0.0
    paired = _paired_effects(run_rows, DECISION_TARGETED_POLICY)
    assessment = _assessment(
        paired, run_rows, True, config, DECISION_TARGETED_POLICY
    )
    corrected_decisions = dict(summary["protocol_gate_decisions"])
    corrected_decisions.update({key: True for key in LEGACY_FALSE_KEYS})
    if not all(corrected_decisions.values()):
        raise AssertionError("P3L.2 corrected protocol gate did not close")

    artifacts = {
        str(path.relative_to(root)).replace("\\", "/"): _hash(path)
        for path in required
    }
    return {
        "schema": SCHEMA,
        "stage": "P3L.2",
        "status": assessment["status"],
        "correction_scope": (
            "reporting-semantics-only-no-rescoring-no-selection-no-response-access"
        ),
        "original_protocol_status": summary["effectiveness_assessment"]["status"],
        "original_false_protocol_keys": sorted(false_keys),
        "corrected_protocol_gate_passed": True,
        "corrected_protocol_gate_decisions": corrected_decisions,
        "corrected_effectiveness_assessment": assessment,
        "formal_protocol_evidence": True,
        "formal_efficacy_evidence": bool(assessment["strong_evidence"]),
        "real_measurement_experiment": True,
        "heldout_opened": False,
        "selection_recomputed": False,
        "responses_reopened": False,
        "successful_runs": len(run_rows),
        "expected_runs": 96,
        "query_rows": len(queries),
        "pcpi_query_rows": len(pcpi_queries),
        "corrected_query_check_rates": check_rates,
        "paired_effects": paired,
        "evidence_registry": {
            "valid": True,
            "event_count": len(EvidenceRegistry(registry_path).events()),
            "head_hash": manifest["evidence_registry"]["head_hash"],
        },
        "source_artifact_hashes": artifacts,
        "experiment_source_commit": EXPERIMENT_COMMIT,
        "experiment_source_tree": EXPERIMENT_TREE,
        "auditor_source_commit": _git_value("rev-parse", "HEAD"),
        "auditor_source_tree": _git_value("rev-parse", "HEAD^{tree}"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("P3L.2 audit correction output already exists")
    result = audit_completed_run(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "status": result["status"],
        "corrected_protocol_gate_passed": True,
        "formal_efficacy_evidence": result["formal_efficacy_evidence"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
