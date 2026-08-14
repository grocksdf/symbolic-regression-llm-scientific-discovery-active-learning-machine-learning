"""Shared symbolic-search contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateFormula:
    expression: str
    source: str
    score: float | None = None
    lineage_id: str = ""


__all__ = ["CandidateFormula"]
