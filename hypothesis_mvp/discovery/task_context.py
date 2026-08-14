"""Leakage-aware scientific task context for discovery prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping


_ASSIGNMENT = re.compile(r"(?:^|\s)[A-Za-z_][A-Za-z0-9_]*(?:\([^\n)]*\))?\s*=")
_FORBIDDEN_HINTS = (
    "ground truth", "target equation", "true equation", "answer:",
    "sympy_format", "lambda_format", "program_format",
)


def _safe_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    lowered = text.lower()
    if any(marker in lowered for marker in _FORBIDDEN_HINTS) or _ASSIGNMENT.search(text):
        return ""
    return text[:limit]


@dataclass(frozen=True)
class DiscoveryTaskContext:
    """Scientific semantics allowed on the structure-generation surface.

    Dataset identifiers, target expressions, answers and held-out observations
    are deliberately absent. Text resembling an equation assignment or an
    explicit answer is removed before it reaches an LLM prompt.
    """

    description: str = ""
    variable_descriptions: Mapping[str, str] = field(default_factory=dict)
    domain: str = "general_scientific_system"
    source: str = "caller_declared_selection_context"

    def prompt_payload(self, n_features: int) -> dict[str, Any]:
        variables = {
            f"x{index}": _safe_text(
                self.variable_descriptions.get(f"x{index}", ""), limit=320
            )
            for index in range(int(n_features))
        }
        variables = {key: value for key, value in variables.items() if value}
        description = _safe_text(self.description, limit=2400)
        domain = _safe_text(self.domain, limit=160) or "general_scientific_system"
        return {
            "name": "scientific_equation_discovery",
            "description": description or "Infer an executable equation from measured variables.",
            "domain": domain,
            "variables": variables,
        }

    def audit(self, n_features: int) -> dict[str, Any]:
        payload = self.prompt_payload(n_features)
        rendered = repr(sorted(payload.items())).encode("utf-8")
        return {
            "schema": "discovery-task-context-v1",
            "source": self.source,
            "prompt_context_sha256": hashlib.sha256(rendered).hexdigest(),
            "description_included": bool(_safe_text(self.description, limit=2400)),
            "variable_description_count": len(payload["variables"]),
            "dataset_identifier_in_prompt": False,
            "target_equation_in_prompt": False,
            "heldout_observations_in_prompt": False,
        }


__all__ = ["DiscoveryTaskContext"]
