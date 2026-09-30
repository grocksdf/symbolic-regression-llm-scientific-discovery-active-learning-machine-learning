"""Static contract tests for the dedicated benchmark runtime Gate."""

from scripts.run_aistats_drr_runtime_gate import PACKAGES


def test_runtime_gate_requires_benchmark_and_scientific_dependencies():
    assert {"datasets", "h5py", "PyYAML", "numpy", "scipy"} <= set(PACKAGES)
