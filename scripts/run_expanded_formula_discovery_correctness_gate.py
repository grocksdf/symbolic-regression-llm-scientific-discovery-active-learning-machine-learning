"""No-data correctness and leakage Gate for Expanded Formula Discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from scripts.expanded_formula_discovery_harness import (
    ADMISSION_ARMS, exact_formula_recovery, project_admission_arms,
    structural_formula_recovery,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    def role(kind, start):
        X = np.linspace(start, start + 1, 16)[:, None]
        return RoleDataset(kind, X, X[:, 0] ** 2 + .1 * X[:, 0])
    core = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
    ]
    optional = [{
        "expression": "x0**2", "source": "llm", "origin": "llm"}]
    projection = project_admission_arms(
        core, optional, role(DataRole.DEVELOPMENT, 0),
        role(DataRole.VALIDATION, 2),
        role(DataRole.VALIDATION, 4),
        role(DataRole.VALIDATION, 6),
        np.asarray([[0.], [1.], [2.]]),
        n_features=1, identity="b" * 64)
    overlap_failed_closed = False
    reused = role(DataRole.VALIDATION, 2)
    try:
        project_admission_arms(
            core, optional, role(DataRole.DEVELOPMENT, 0),
            reused, reused, role(DataRole.VALIDATION, 6),
            np.asarray([[0.], [1.], [2.]]),
            n_features=1, identity="c" * 64)
    except ValueError:
        overlap_failed_closed = True
    runner_source = (
        ROOT / "scripts/run_expanded_formula_discovery.py"
    ).read_text(encoding="utf-8")
    child_source = (
        ROOT / "scripts/run_aistats_three_arm_child.py"
    ).read_text(encoding="utf-8")
    checks = {
        "three_admission_arms_registered": ADMISSION_ARMS == (
            "independent_protected", "same_data", "accept_all"),
        "exact_equivalence_positive_control":
            exact_formula_recovery("2*x0 + 1", "1 + 2*t", ["t"]),
        "exact_equivalence_negative_control":
            not exact_formula_recovery("2*x0", "1 + 2*t", ["t"]),
        "structural_positive_control": structural_formula_recovery(
            "3*x0**2 + 7*sin(x0)", "9*t**2 + 2*sin(t)", ["t"]),
        "structural_negative_control": not structural_formula_recovery(
            "3*x0 + 7*sin(x0)", "9*t**2 + 2*sin(t)", ["t"]),
        "one_frozen_candidate_bank_identity": bool(
            projection["candidate_bank_identity"]),
        "complete_core_is_protected_in_every_arm": all(
            rows[:2] == core for rows in projection["arms"].values()),
        "accept_all_uses_no_admission_response":
            projection["audits"]["accept_all"]["response_accessed"] is False,
        "llm_has_no_epistemic_authority":
            projection[
                "generative_authority_equals_epistemic_authority"] is False,
        "overlapping_gap_and_admission_roles_fail_closed":
            overlap_failed_closed,
        "ground_truth_opens_after_hash_bound_admission_completion": (
            runner_source.index("admission completion identity failed")
            < runner_source.index("metadata = _metadata_rows(")),
        "generation_metadata_excludes_ground_truth_expression": (
            'columns = ["name", "symbols", "symbol_descs", '
            '"symbol_properties"]' in child_source
            and '"ground_truth_expression_opened": False' in child_source),
        "physics_family_generation_registered":
            '"phys_osc": "lsr_synth/phys_osc"' in child_source,
    }
    result = {
        "schema":
            "scientific-expanded-formula-discovery-correctness-gate-v1",
        "passed": all(checks.values()), "checks": checks,
        "real_data_accessed": False, "candidate_bank_accessed": False,
        "ground_truth_expression_values_opened": False,
        "test_or_ood_accessed": False,
        "confirmation_responses_opened": False,
        "confirmation_results_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "Synthetic evaluator and admission-interface correctness only; "
            "no formula discovery or efficacy evidence."),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
