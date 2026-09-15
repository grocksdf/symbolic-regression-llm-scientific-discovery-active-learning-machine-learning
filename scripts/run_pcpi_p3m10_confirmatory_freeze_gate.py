"""No-data gate for the resolution-stratified confirmatory claim."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3m_10_resolution_stratified_confirmatory.json"

def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "schema_registered": cfg.get("schema") == "pcpi-p3m10-resolution-stratified-confirmatory-config-v1",
        "primary_estimand_registered": cfg.get("primary_estimand") == "frozen-operational-class-decision-risk-on-identifiable-families",
        "identifiability_rule_registered": cfg.get("identifiability_rule") == "registered-resolution-class-aggregation-positive",
        "non_identifiable_negative_evidence": cfg.get("non_identifiable_family_role", "").startswith("report-negative-evidence"),
        "secondary_all_families_registered": cfg.get("secondary_estimand", "").endswith("all-families"),
        "heldout_closed": cfg.get("heldout_state") == "closed",
        "no_threshold_mutation": cfg.get("threshold_mutation_allowed") is False,
        "no_family_deletion": cfg.get("family_deletion_allowed") is False,
        "no_posthoc_reclassification": cfg.get("posthoc_family_reclassification_allowed") is False,
        "confirmatory_authorization_registered": cfg.get("confirmatory_execution_authorized") is True,
        "runner_identity_registered": cfg.get("checkpoint_schema") == "pcpi-p3m9-downside-severity-risk-checkpoint-v1",
    }
    report = {
        "schema": "pcpi-p3m10-resolution-stratified-freeze-gate-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "real_data_access": False,
        "heldout_access": False,
        "confirmatory_execution_authorized": False,
        "claim_boundary": "Conditional class-risk superiority only for families identifiable at the frozen registered resolution; non-identifiable families remain negative evidence. Predictive AULC is secondary across all families.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
