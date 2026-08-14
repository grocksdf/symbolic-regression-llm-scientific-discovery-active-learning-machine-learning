import csv
from dataclasses import fields
from pathlib import Path

from hypothesis_mvp.symbolic.contracts import CandidateFormula


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_p0_required_artifacts_exist() -> None:
    required = {
        "pcpi_mathematical_contract.md",
        "pcpi_claim_code_evidence_matrix.csv",
        "pcpi_smc_correctness_audit.md",
        "pcpi_data_leakage_audit.md",
        "pcpi_gap_and_rebuild_plan.md",
        "P0_DECISION.md",
    }
    assert required <= {path.name for path in DOCS.iterdir() if path.is_file()}


def test_mathematical_contract_freezes_required_decisions() -> None:
    text = (DOCS / "pcpi_mathematical_contract.md").read_text(encoding="utf-8")
    required_phrases = (
        "homoscedastic Gaussian likelihood",
        "InverseGamma",
        "finite-action operational equivalence",
        "class information gain",
        "default primary loss is 0–1 loss",
        "provenance-verified measured observations",
    )
    for phrase in required_phrases:
        assert phrase in text


def test_claim_matrix_has_unique_claim_ids_and_boundaries() -> None:
    path = DOCS / "pcpi_claim_code_evidence_matrix.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    identifiers = [row["claim_id"] for row in rows]
    assert len(identifiers) == len(set(identifiers))
    assert all(row["current_status"] and row["allowed_wording"] for row in rows)
    assert {"C2_SMC", "C3_ACQUISITION", "PAPER_DRAFT"} <= set(identifiers)


def test_heuristic_candidate_contract_has_no_false_posterior_field() -> None:
    assert "posterior_prob" not in {field.name for field in fields(CandidateFormula)}
