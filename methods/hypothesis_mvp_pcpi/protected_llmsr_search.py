"""Protected adapter for the official LLM-SR multi-island search buffer.

The repository's ``methods/llmsr/buffer.py`` is the Apache-2.0 DeepMind
FunSearch-style implementation used by LLM-SR.  This adapter reuses its
tempered softmax sampling rule while storing only response-free candidate
identities plus development-stage scores.  Admission/update/reporting values
are never accepted by this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import scipy.special


def _official_llmsr_softmax(logits, temperature):
    """Adapted from methods/llmsr/buffer.py (Apache-2.0)."""
    values = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("LLM-SR island logits must be finite")
    result = scipy.special.softmax(values / float(temperature), axis=-1)
    index = int(np.argmax(result))
    result[index] = (
        1. - np.sum(result[:index]) - np.sum(result[index + 1:]))
    return result


@dataclass(frozen=True)
class ProtectedSearchCandidate:
    expression: str
    lineage_id: str
    score: float
    signature: tuple[float, ...]
    source_condition: str
    generation_round: int

    @property
    def identity(self):
        return sha256(json.dumps({
            "expression": self.expression,
            "lineage_id": self.lineage_id,
            "source_condition": self.source_condition,
            "generation_round": self.generation_round,
        }, sort_keys=True).encode()).hexdigest()

    def prompt_row(self):
        return {
            "entry_id": self.identity,
            "expression": self.expression,
            "development_score": float(self.score),
            "development_signature": list(self.signature),
            "lineage_id": self.lineage_id,
            "source_condition": self.source_condition,
            "generation_round": self.generation_round,
            "response_values_retained": False,
        }


class ProtectedLLMSRBuffer:
    """Deterministic short-horizon multi-island candidate buffer."""

    def __init__(
        self, *, num_islands: int, functions_per_prompt: int,
        temperature: float, seed: int,
    ):
        if (type(num_islands) is not int or num_islands < 2
                or type(functions_per_prompt) is not int
                or functions_per_prompt < 1
                or not 0 < temperature <= 1):
            raise ValueError("invalid protected LLM-SR buffer configuration")
        self.num_islands = num_islands
        self.functions_per_prompt = functions_per_prompt
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self._islands = [dict() for _ in range(num_islands)]
        self._seen = set()

    def register(
        self, candidate: ProtectedSearchCandidate,
        *, island: int | None = None,
    ):
        if (not candidate.expression or not candidate.lineage_id
                or not np.isfinite(candidate.score)
                or not candidate.signature
                or not np.all(np.isfinite(candidate.signature))):
            raise ValueError("invalid protected search candidate")
        if candidate.identity in self._seen:
            return False
        self._seen.add(candidate.identity)
        targets = (
            range(self.num_islands) if island is None else (int(island),))
        for target in targets:
            if not 0 <= target < self.num_islands:
                raise ValueError("invalid island")
            key = tuple(float(value) for value in candidate.signature)
            current = self._islands[target].get(key)
            if current is None or (
                    candidate.score > current.score
                    or (candidate.score == current.score
                        and candidate.identity < current.identity)):
                self._islands[target][key] = candidate
        return True

    def sample_context(self, island: int | None = None):
        selected_island = (
            int(self.rng.integers(self.num_islands))
            if island is None else int(island))
        clusters = self._islands[selected_island]
        if not clusters:
            return selected_island, []
        candidates = list(clusters.values())
        scores = np.asarray([row.score for row in candidates], dtype=float)
        probabilities = _official_llmsr_softmax(
            scores, self.temperature)
        count = min(self.functions_per_prompt, len(candidates))
        indices = self.rng.choice(
            len(candidates), size=count, replace=False, p=probabilities)
        rows = sorted(
            (candidates[int(index)] for index in indices),
            key=lambda row: (row.score, row.identity))
        return selected_island, [row.prompt_row() for row in rows]

    def reset_weak_half(self):
        best = [
            max((row.score for row in island.values()), default=-np.inf)
            for island in self._islands]
        order = np.argsort(np.asarray(best, dtype=float))
        weak = order[:self.num_islands // 2]
        strong = order[self.num_islands // 2:]
        if not len(strong) or not np.isfinite(np.asarray(best)[strong]).any():
            return []
        resets = []
        for target in weak:
            founders = [
                row for index in strong
                for row in self._islands[int(index)].values()]
            if not founders:
                continue
            founder = max(
                founders, key=lambda row: (row.score, -len(row.expression)))
            self._islands[int(target)] = {
                tuple(founder.signature): founder}
            resets.append({
                "reset_island": int(target),
                "founder_identity": founder.identity})
        return resets

    def audit(self):
        return {
            "schema": "protected-llmsr-buffer-v1",
            "num_islands": self.num_islands,
            "functions_per_prompt": self.functions_per_prompt,
            "temperature": self.temperature,
            "unique_candidate_count": len(self._seen),
            "cluster_count_by_island": [
                len(island) for island in self._islands],
            "admission_response_accessed": False,
            "posterior_update_response_accessed": False,
            "reporting_response_accessed": False,
            "heldout_opened": False,
        }


__all__ = ["ProtectedLLMSRBuffer", "ProtectedSearchCandidate"]
