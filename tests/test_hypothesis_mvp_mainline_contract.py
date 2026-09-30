from __future__ import annotations

import inspect
from pathlib import Path

from methods.hypothesis_mvp_pcpi._mainline import load
from methods.hypothesis_mvp_pcpi.drr_searcher import DRRBenchmarkSearcher
from methods.hypothesis_mvp_pcpi.searcher import PCPISearcher


ROOT = Path(__file__).resolve().parents[1]


def test_mainline_exports_the_only_adapter_contract() -> None:
    discovery = load("")
    assert callable(discovery.discover_from_selection)
    assert callable(load("drr_readiness").evaluate_drr_readiness)
    parameters = inspect.signature(discovery.discover_from_selection).parameters
    assert tuple(parameters) == ("selection", "kwargs")
    for removed in (
            "DiscoveryStores", "ConfirmationEvidence", "discover_from_arrays",
            "record_confirmation_evidence", "promote_staged_knowledge"):
        assert not hasattr(discovery, removed)


def test_paper_profile_locks_published_experiment_constants() -> None:
    text = (ROOT / "configs" / "restart-paper.yaml").read_text(encoding="utf-8")
    assert "profile_name: restart-paper-iclr2026" in text
    assert "evaluation_budget: 1000" in text
    assert "candidates_per_island: 4" in text
    assert "exemplars_per_prompt: 4" in text
    assert sum(line.strip().startswith("- island_") for line in text.splitlines()) == 10
    assert "structure_library_max_entries: 20" in text


def test_active_adapter_contains_no_scientific_runtime_copy() -> None:
    active = ROOT / "methods" / "hypothesis_mvp_pcpi"
    assert sorted(path.name for path in active.glob("*.py")) == [
        "__init__.py", "_mainline.py", "drr_adapter.py",
        "drr_searcher.py", "quality_first_augmentation.py", "searcher.py",
    ]
    assert PCPISearcher is not None and DRRBenchmarkSearcher is not None
