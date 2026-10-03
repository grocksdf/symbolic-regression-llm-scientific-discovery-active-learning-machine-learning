from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v29_restores_stable_serial_scheduling():
 c=(ROOT/"configs/aistats_three_arm_formula_prospective.yaml").read_text()
 r=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
 b=(ROOT/"scripts/build_expanded_formula_v29_continuation.py").read_text()
 assert "engine_workers: 1" in c
 assert '"task_parallelism":1' in b
 assert '"parallel_engine_teardown_rejected":True' in b
 assert "if parallelism == 1:" in r
 assert "scientific-expanded-formula-discovery-freeze-v2.9" in r
