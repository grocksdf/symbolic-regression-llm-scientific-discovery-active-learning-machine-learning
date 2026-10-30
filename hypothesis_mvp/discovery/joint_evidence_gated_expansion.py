"""Fixed-batch Joint Evidence-Gated Posterior Expansion (JEGPE).

This module deliberately does not implement an anytime-valid expansion loop.
One candidate archive is frozen, one bundle is selected without admission
responses, and that one bundle is audited as a fixed alternative against the
protected core predictive law.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior

from .candidate_region_expansion import FrozenAxisRegions
from .pcpi_adapter import support_parser_for_policy
from .regional_candidate_admission import (
    CLOSED_POLICY,
    _fixed_predictive_logpdf,
    global_protected_evidence_decision,
)


@dataclass(frozen=True)
class ArchiveCandidate:
    expression: str
    source: str
    lineage_id: str
    support: tuple[str, ...]
    structural_hash: str
    complexity: int

    def posterior_row(self) -> dict:
        return {
            "expression": self.expression,
            "source": self.source,
            "origin": "archive",
            "lineage_id": self.lineage_id,
        }

    def generator_view(self) -> dict:
        """Training-only view: never include admission or reporting fields."""
        return {
            "expression": self.expression,
            "source": self.source,
            "lineage_id": self.lineage_id,
            "structural_hash": self.structural_hash,
            "complexity": self.complexity,
            "support": list(self.support),
        }


@dataclass(frozen=True)
class CandidateArchive:
    candidates: tuple[ArchiveCandidate, ...]
    n_features: int
    coefficient_policy: str
    identity: str

    @classmethod
    def freeze(
        cls, rows: Iterable[Mapping],
        *, n_features: int,
        coefficient_policy: str = CLOSED_POLICY,
    ) -> "CandidateArchive":
        parser = support_parser_for_policy(coefficient_policy)
        unique = {}
        for raw in rows:
            expression = str(raw.get("expression") or "").strip()
            if not expression:
                continue
            support = tuple(parser(expression, n_features))
            structural_hash = sha256(json.dumps(
                support, sort_keys=True).encode()).hexdigest()
            candidate = ArchiveCandidate(
                expression=expression,
                source=str(raw.get("source") or "unknown"),
                lineage_id=str(raw.get("lineage_id") or ""),
                support=support,
                structural_hash=structural_hash,
                complexity=sum(term.count("formula_ast_v1:") + 1
                               for term in support),
            )
            previous = unique.get(structural_hash)
            if previous is None or (
                    candidate.complexity, candidate.expression,
                    candidate.source) < (
                    previous.complexity, previous.expression,
                    previous.source):
                unique[structural_hash] = candidate
        candidates = tuple(sorted(
            unique.values(),
            key=lambda row: (row.structural_hash, row.expression)))
        material = {
            "n_features": n_features,
            "coefficient_policy": coefficient_policy,
            "candidates": [row.generator_view() for row in candidates],
        }
        return cls(
            candidates, n_features, coefficient_policy,
            sha256(json.dumps(
                material, sort_keys=True).encode()).hexdigest())

    def generator_view(self) -> list[dict]:
        return [candidate.generator_view() for candidate in self.candidates]

    def bundles(self, maximum_size: int) -> tuple[tuple[ArchiveCandidate, ...], ...]:
        if type(maximum_size) is not int or maximum_size < 1:
            raise ValueError("maximum bundle size must be positive")
        return tuple(
            bundle
            for size in range(1, min(maximum_size, len(self.candidates)) + 1)
            for bundle in combinations(self.candidates, size))


def bundle_identity(bundle: Sequence[ArchiveCandidate]) -> str:
    return sha256(json.dumps(
        [candidate.structural_hash for candidate in bundle],
        sort_keys=True).encode()).hexdigest()


def support_redundancy(bundle: Sequence[ArchiveCandidate]) -> float:
    if len(bundle) < 2:
        return 0.0
    values = []
    for left, right in combinations(bundle, 2):
        a, b = set(left.support), set(right.support)
        values.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(values))


def select_fixed_budget_bundle(
    archive: CandidateArchive,
    score: Callable[[tuple[ArchiveCandidate, ...]], float],
    *, maximum_size: int,
    evaluation_budget: int,
) -> tuple[tuple[ArchiveCandidate, ...], dict]:
    """Select by full-bank score; ties prefer smaller, simpler, diverse sets."""
    bundles = archive.bundles(maximum_size)
    if type(evaluation_budget) is not int or evaluation_budget < 1:
        raise ValueError("joint selection needs a positive evaluation budget")
    ranked = sorted(bundles, key=bundle_identity)[:evaluation_budget]
    records = []
    for bundle in ranked:
        value = float(score(bundle))
        if not np.isfinite(value):
            raise ValueError("joint selection score is nonfinite")
        records.append({
            "bundle": bundle,
            "identity": bundle_identity(bundle),
            "score": value,
            "size": len(bundle),
            "complexity": sum(row.complexity for row in bundle),
            "redundancy": support_redundancy(bundle),
        })
    if not records:
        return (), {
            "schema": "jegpe-training-bundle-selection-v1",
            "selected_identity": None, "selected_score": 0.0,
            "evaluated_bundle_count": 0, "records": []}
    best = min(records, key=lambda row: (
        -row["score"], row["size"], row["complexity"],
        row["redundancy"], row["identity"]))
    selected = best["bundle"] if best["score"] > 0.0 else ()
    audit_records = [{
        key: value for key, value in row.items() if key != "bundle"}
        for row in records]
    return selected, {
        "schema": "jegpe-training-bundle-selection-v1",
        "selected_identity": (
            bundle_identity(selected) if selected else None),
        "selected_score": float(best["score"]) if selected else 0.0,
        "selected_size": len(selected),
        "maximum_size": maximum_size,
        "evaluation_budget": evaluation_budget,
        "evaluated_bundle_count": len(records),
        "records": audit_records,
        "admission_response_accessed": False,
        "reporting_response_accessed": False,
    }


def extract_subexpression_materials(
    archive: CandidateArchive,
    *, maximum_per_candidate: int = 32,
) -> list[dict]:
    """Extract canonical training-side AST materials without validation labels."""
    if maximum_per_candidate < 1:
        raise ValueError("subexpression budget must be positive")
    output, seen = [], set()
    parser = support_parser_for_policy(archive.coefficient_policy)
    for candidate in archive.candidates:
        root = ast.parse(candidate.expression, mode="eval")
        count = 0
        for node in ast.walk(root):
            if not isinstance(node, (
                    ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
                continue
            expression = ast.unparse(node)
            try:
                support = tuple(parser(expression, archive.n_features))
            except (SyntaxError, TypeError, ValueError):
                continue
            identity = sha256(json.dumps(
                support, sort_keys=True).encode()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            output.append({
                "expression": expression,
                "structural_hash": identity,
                "parent_structural_hash": candidate.structural_hash,
                "parent_lineage_id": candidate.lineage_id,
                "source": candidate.source,
                "support": list(support),
            })
            count += 1
            if count >= maximum_per_candidate:
                break
    return output


def admit_joint_bundle(
    core_rows: Sequence[Mapping],
    bundle: Sequence[ArchiveCandidate],
    fit: RoleDataset, gap_audit: RoleDataset,
    admission: RoleDataset, domain: np.ndarray,
    regions: FrozenAxisRegions, eligible_regions: tuple[int, ...],
    prior: NormalInverseGammaPrior, measurement_budget: int,
    *, alpha: float = .05,
    coefficient_policy: str = CLOSED_POLICY,
    tested_bundle_count: int = 1,
) -> tuple[list[dict], dict]:
    """Audit one frozen expanded bank against the protected core bank."""
    if (fit.role is not DataRole.DEVELOPMENT
            or gap_audit.role is not DataRole.VALIDATION
            or admission.role is not DataRole.VALIDATION
            or fit.row_fingerprints & (
                gap_audit.row_fingerprints | admission.row_fingerprints)
            or gap_audit.row_fingerprints & admission.row_fingerprints
            or not 0 < alpha < 1
            or type(tested_bundle_count) is not int
            or tested_bundle_count < 1
            or regions.n_features != fit.X.shape[1]):
        raise ValueError(
            "joint admission needs frozen bundle and disjoint registered roles")
    core = [dict(row) for row in core_rows]
    expanded_rows = [candidate.posterior_row() for candidate in bundle]
    if not core or not expanded_rows:
        return [], {
            "schema": "candidate-joint-global-protected-admission-v1",
            "passed": False, "reason": "empty-core-or-bundle",
            "bundle_size": len(expanded_rows),
            "candidate_response_accessed": False,
            "heldout_opened": False}
    parser = support_parser_for_policy(coefficient_policy)
    core_supports = {
        parser(row["expression"], fit.X.shape[1]) for row in core}
    bundle_supports = [
        parser(row["expression"], fit.X.shape[1])
        for row in expanded_rows]
    if (len(set(bundle_supports)) != len(bundle_supports)
            or any(support in core_supports for support in bundle_supports)):
        raise ValueError("joint bundle duplicates core or itself")
    core_score, core_identity = _fixed_predictive_logpdf(
        core, fit, admission, domain, prior, measurement_budget,
        coefficient_policy=coefficient_policy)
    expanded_score, expanded_identity = _fixed_predictive_logpdf(
        [*core, *expanded_rows], fit, admission, domain,
        prior, measurement_budget,
        coefficient_policy=coefficient_policy)
    threshold = math.log(tested_bundle_count / alpha)
    decision = global_protected_evidence_decision(
        expanded_score - core_score, regions.assign(admission.X),
        eligible_regions, threshold)
    return (expanded_rows if decision["passed"] else []), {
        "schema": "candidate-joint-global-protected-admission-v1",
        **decision,
        "bundle_identity": bundle_identity(bundle),
        "bundle_size": len(bundle),
        "bundle_structural_hashes": [
            candidate.structural_hash for candidate in bundle],
        "core_identity": core_identity,
        "expanded_identity": expanded_identity,
        "gap_identity": gap_audit.fingerprint,
        "admission_identity": admission.fingerprint,
        "tested_bundle_count": tested_bundle_count,
        "decision_rule": (
            "one frozen expanded-bank predictive likelihood ratio above "
            "M/alpha and nonnegative cumulative evidence in every region"),
        "fixed_batch_only": True,
        "anytime_valid_claimed": False,
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }
