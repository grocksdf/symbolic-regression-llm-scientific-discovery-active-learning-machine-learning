"""Static checks for the offline LLM admission audit."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT / "scripts/build_aistats_llm_admission_offline_audit.py"
).read_text(encoding="utf-8")


def test_admission_audit_is_read_only_and_offline() -> None:
    assert '"read_only": True' in AUDIT
    assert '"rerun_authorized": False' in AUDIT
    assert '"efficacy_demonstrated": False' in AUDIT
    assert "offline-read-only-audit" in AUDIT
    assert "raise ValueError(\"LLM admission audit output must be new\")" in AUDIT


def test_admission_audit_declares_inner_llm_confound() -> None:
    # These runs cannot evidence free-form LLM equation proposal, so the
    # certificate must say so instead of implying an LLM efficacy result.
    assert "TYPED_EVIDENCE_SYNTHESIS_PROFILE = True" in AUDIT
    assert '"inner_llm_equation_proposal_active": False' in AUDIT
    assert "inner_llm_path_confound_declared" in AUDIT
    assert "typed evidence-synthesis profile the inner equation-proposal LLM" in AUDIT
    assert "free-form LLM equation proposals" in AUDIT


def test_admission_audit_never_imputes_missing_runs() -> None:
    assert "missing_full_coordinates" in AUDIT
    assert "unexpected_full_coordinates" in AUDIT
    assert "missing_runs_declared_not_imputed" in AUDIT
    assert "not a rerun, not a rescoring" in AUDIT


def test_admission_audit_counts_reconcile() -> None:
    assert "directive_counts_reconcile" in AUDIT
    assert "compiled_never_exceeds_passed" in AUDIT
    assert "synthesis_final_topk_rows_total" in AUDIT
    assert "same_best_expression" in AUDIT


def test_admission_audit_normalizes_counters_to_plain_int() -> None:
    # numpy scalars cannot be JSON encoded; every counter must be normalized.
    assert "def _count(value: object) -> int:" in AUDIT
    assert "json.loads(payload)" in AUDIT
    assert "_counter_pair" in AUDIT


def test_admission_audit_keeps_discovery_roles_closed() -> None:
    assert "discovery_roles_closed" in AUDIT
    assert '"candidate_response_accessed": False' in AUDIT
    assert '"test_or_ood_accessed": False' in AUDIT
    assert '"heldout_opened": False' in AUDIT
