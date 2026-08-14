from pathlib import Path

from hypothesis_mvp.hypotheses import (
    dependency_specification_hash,
    runtime_dependency_hash,
    runtime_dependency_snapshot,
)
from scripts.audit_final_source import audit


ROOT = Path(__file__).resolve().parents[1]


def test_final_source_integrity() -> None:
    report = audit()
    assert report["status"] == "passed", report["failures"]


def test_formal_dependency_identity_records_exact_runtime_versions() -> None:
    first = runtime_dependency_snapshot()
    second = runtime_dependency_snapshot()
    assert first == second
    assert runtime_dependency_hash(first) == runtime_dependency_hash(second)
    assert first["python"]["version"]
    assert first["platform"]["machine"]
    for name in ("numpy", "scipy", "pytest"):
        assert first["distributions"][name]


def test_dependency_specification_hash_is_deterministic_and_complete() -> None:
    first = dependency_specification_hash(ROOT)
    second = dependency_specification_hash(ROOT)
    assert first == second
    assert len(first) == 64
