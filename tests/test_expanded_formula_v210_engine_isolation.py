from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v210_enables_killable_engine_deadlines():
 r=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
 b=(ROOT/"scripts/build_expanded_formula_v210_continuation.py").read_text()
 assert 'env["FORMULA_ENGINE_PROCESS_ISOLATION"] = "1"' in r
 assert '"engine_timeout_is_hard_kill":True' in b
 assert "scientific-expanded-formula-discovery-freeze-v2.10" in r
