from pathlib import Path
import re


def test_no_answer_metadata_enters_production_source() -> None:
    root = Path(__file__).resolve().parents[1] / "hypothesis_mvp"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in root.rglob("*.py")
    )
    for forbidden in (
        "oracle_expr",
        "ground_truth_expression",
        "target_formula",
        "metadata_hint",
    ):
        assert forbidden not in source
    assert not re.search(
        r"metadata\s*(?:\[|\.get\()[^\n]*['\"](?:expression|oracle)", source
    )
