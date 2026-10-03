from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_v27_outer_wall_exceeds_provider_exhaustion_path():
    b=(ROOT/"scripts/build_expanded_formula_v27_continuation.py").read_text()
    r=(ROOT/"scripts/run_expanded_formula_discovery_prospective.py").read_text()
    assert '"wall_time_seconds_per_run":1800' in b
    assert '"outer_exceeds_one_exhausted_request_path":True' in b
    assert '"task_parallelism":2' in b
    assert '"provider_concurrency":1' in b
    assert '"engine_workers":4' in b
    assert '"provider_lock_relative_path":"FORMULA_PROVIDER.lock"' in b
    assert "scientific-expanded-formula-discovery-freeze-v2.7" in r
    assert "ThreadPoolExecutor(max_workers=parallelism)" in r
    assert 'env["FORMULA_PROVIDER_LOCK"]' in r
