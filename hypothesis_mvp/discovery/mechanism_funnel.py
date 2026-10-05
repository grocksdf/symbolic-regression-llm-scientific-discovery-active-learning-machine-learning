"""Prospective mechanism accounting, never an efficacy success criterion."""
from collections import Counter
import ast


def _fraction(numerator, denominator):
    if not 0 <= numerator <= denominator:
        raise ValueError("inconsistent mechanism numerator/denominator")
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _family_signature(expression):
    operators = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
                 ast.Div: "div", ast.Pow: "pow"}
    tree = ast.parse(expression, mode="eval")
    labels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            labels.add(operators.get(type(node.op), "other"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            labels.add(node.func.id)
    return "+".join(sorted(labels)) or "linear-or-constant"


def candidate_mechanism_funnel(report, synthesis_audit, admission_report=None):
    attempts = [row for row in report.get("candidate_validity_attempts", ())
                if row.get("origin") == "llm" and not row.get("reused")]
    typed = [row for row in attempts if row.get("source") == "llm_evidence_synthesis"]
    compiled = [row for row in attempts if row.get("compiled")]
    transition_rows = [row for batch in report.get("llm_rounds", ())
                for island in batch.get("islands", ())
                for row in island.get("candidate_audit", ()) if row.get("validated")]
    accepted = [row for row in transition_rows if row.get("accepted")]
    admission = dict(admission_report or {})
    admission_rows = admission.get("candidates", ())
    admission_evaluated = [row for row in admission_rows
                           if "region_log_predictive_ratios" in row]
    supports = {tuple(row.get("structural_support", ())) for row in compiled
                if row.get("domain_valid") and row.get("structural_support")}
    families = Counter(_family_signature(row["expression"]) for row in compiled
                       if row.get("domain_valid") and row.get("expression"))
    return {"schema": "candidate-mechanism-funnel-v1",
            "population": "new-LLM-candidate-evaluation-attempts",
            "typed_directives": synthesis_audit.get("directive_count", 0),
            "typed_compiled_proposals": synthesis_audit.get("compiled_candidate_count", 0),
            "typed_preflight_attempts": len(typed),
            "domain_valid_given_typed": _fraction(
                sum(bool(row.get("domain_valid")) for row in typed), len(typed)),
            "fit_survives_given_compiled": _fraction(
                sum(bool(row.get("fit_survived")) for row in compiled), len(compiled)),
            "discovery_accepted_given_evaluated": _fraction(
                len(accepted), len(transition_rows)),
            "discovery_population": "validated-inner-incumbent-transition-proposals",
            "independent_admitted_given_admission_evaluated": _fraction(
                sum(bool(row.get("admitted")) for row in admission_evaluated),
                len(admission_evaluated)),
            "independent_attempts": admission.get("attempts", 0),
            "distinct_valid_support_count": len(supports),
            "valid_operator_family_counts": dict(sorted(families.items())),
            "failure_counts": dict(Counter(row.get("rejection_reason") for row in attempts
                                           if row.get("rejection_reason"))),
            "reused_candidates_excluded": True,
            "efficacy_claim_authorized": False}
