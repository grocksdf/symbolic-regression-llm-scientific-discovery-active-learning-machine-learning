from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v28_windows_lock_waits_without_provider_retry():
 child=(ROOT/"scripts/run_aistats_three_arm_formula_child.py").read_text()
 runner=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
 builder=(ROOT/"scripts/build_expanded_formula_v28_continuation.py").read_text()
 assert "msvcrt.LK_NBLCK" in child
 assert "time.sleep(0.1)" in child
 assert 'getattr(error, "errno", None) not in {13, 36}' in child
 assert 'env["FORMULA_PROVIDER_LOCK_TIMEOUT"]' in runner
 assert '"lock_contention_counts_as_provider_attempt":False' in builder
 assert "scientific-expanded-formula-discovery-freeze-v2.8" in runner
