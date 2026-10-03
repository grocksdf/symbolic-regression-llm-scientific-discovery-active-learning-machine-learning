from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v211_registers_no_formula_semantic_abstention():
 r=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
 b=(ROOT/"scripts/build_expanded_formula_v211_continuation.py").read_text()
 assert "scientific-expanded-formula-discovery-freeze-v2.11" in r
 assert '"fallback_formula_generated":False' in b
 assert '"new_scientific_evidence_available":False' in b
 assert "twenty-two reused runs" in b
