"""LLM-SRBench benchmark adapter for the Hypothesis-MVP mainline."""

from __future__ import annotations

from typing import Any

__all__ = ["PCPISearcher"]


def __getattr__(name: str) -> Any:
    if name != "PCPISearcher":
        raise AttributeError(name)
    from .searcher import PCPISearcher
    return PCPISearcher
