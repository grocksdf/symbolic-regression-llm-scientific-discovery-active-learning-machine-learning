"""Convert discovery results into the canonical hypothesis and evidence contracts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    HypothesisSpec,
    HypothesisStatus,
    VariableSpec,
)

from .contracts import json_safe


def array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sequence(metadata: Mapping[str, Any], name: str, count: int, default: str) -> list[str]:
    raw = metadata.get(name)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == count:
        return [str(value).strip() or default for value in raw]
    return [default for _ in range(count)]


def _expression_for_spec(expression: str) -> str:
    value = str(expression).strip().replace("^", "**").replace("Abs(", "abs(")
    return re.sub(r"\bE\b", "e", value)


def build_hypothesis_spec(
    *,
    task_name: str,
    expression: str,
    n_features: int,
    variable_metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    train_fingerprint: str,
    validation_fingerprint: str,
) -> HypothesisSpec:
    metadata = dict(variable_metadata or {})
    names = _sequence(metadata, "feature_names", n_features, "")
    names = [name if name.isidentifier() else f"x{index}" for index, name in enumerate(names)]
    names = [name or f"x{index}" for index, name in enumerate(names)]
    # Runtime equations are expressed in x0..xN. Descriptive source names remain
    # in the descriptions, while executable variable names stay canonical.
    units = _sequence(metadata, "feature_units", n_features, "unknown")
    descriptions = _sequence(metadata, "feature_descriptions", n_features, "")
    variables = tuple(
        VariableSpec(
            name=f"x{index}",
            unit=units[index],
            description=descriptions[index] or f"source variable {names[index]}",
        )
        for index in range(n_features)
    )
    target_name = str(metadata.get("target_name") or "y")
    if not target_name.isidentifier() or target_name in {item.name for item in variables}:
        target_name = "y"
    target = VariableSpec(
        name=target_name,
        unit=str(metadata.get("target_unit") or "unknown"),
        description="measured response",
    )
    normalized_expression = _expression_for_spec(expression)
    identity_material = "|".join(
        (str(task_name), normalized_expression, train_fingerprint, validation_fingerprint)
    )
    hypothesis_id = "hyp-" + sha256(identity_material.encode("utf-8")).hexdigest()[:24]
    metrics = {
        name: json_safe(report.get(name))
        for name in (
            "best_train_nmse",
            "best_val_nmse",
            "best_complexity",
            "best_val_strict_max_relative_error",
            "best_val_relative_error_p99",
            "evaluation_budget_used",
            "llm_call_count",
            "provider_attempt_count",
        )
        if name in report
    }
    spec = HypothesisSpec.create(
        hypothesis_id=hypothesis_id,
        expression=normalized_expression,
        variables=variables,
        target=target,
        domain=str(task_name),
        assumptions=(
            "development and validation rows represent the declared measurement regime",
            "the response is conditionally describable by the declared variables",
            "untouched held-out observations were unavailable to structural selection",
        ),
        mechanism="data-supported mathematical mechanism candidate; physical interpretation remains testable",
        falsifiers=(
            "failure on a preregistered untouched-heldout confirmation set",
            "failure of predicted interventions or regime-specific structural consequences",
        ),
        provenance={
            "runtime": "canonical_scientific_discovery",
            "task_name": str(task_name),
            "train_fingerprint": train_fingerprint,
            "validation_fingerprint": validation_fingerprint,
            "selection_used_heldout": False,
            "metrics": metrics,
            "final_lineage_protocol_valid": bool(report.get("final_lineage_protocol_valid", False)),
            "provider_telemetry": json_safe(report.get("provider_telemetry", [])),
        },
        status=HypothesisStatus.TESTABLE,
    )
    spec.validate_expression()
    return spec


def persist_hypothesis_and_evidence(
    *,
    spec: HypothesisSpec,
    report: Mapping[str, Any],
    hypothesis_dir: str | Path,
    evidence_registry_path: str | Path,
) -> tuple[HypothesisSpec, Path, Path]:
    spec_dir = Path(hypothesis_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{spec.hypothesis_id}.json"
    if spec_path.exists():
        existing = HypothesisSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
        if existing.expression != spec.expression:
            raise RuntimeError(f"hypothesis identity collision: {spec.hypothesis_id}")
        spec = existing
    else:
        spec_path.write_text(
            json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    registry = EvidenceRegistry(evidence_registry_path)
    existing_events = registry.events(hypothesis_id=spec.hypothesis_id)
    if not any(event.event_type is EvidenceEventType.PROPOSED for event in existing_events):
        registry.append(
            hypothesis_id=spec.hypothesis_id,
            event_type=EvidenceEventType.PROPOSED,
            payload={
                "hypothesis_spec_sha256": spec.content_sha256,
                "expression": spec.expression,
                "status": spec.status.value,
                "selection_used_heldout": False,
            },
        )
    validation_fingerprint = str(report.get("validation_fingerprint") or "")
    validation_already_recorded = any(
        event.event_type is EvidenceEventType.TEST_OBSERVED
        and dict(event.payload).get("stage") == "internal_validation"
        and dict(event.payload).get("validation_fingerprint") == validation_fingerprint
        for event in existing_events
    )
    if not validation_already_recorded:
        registry.append(
            hypothesis_id=spec.hypothesis_id,
            event_type=EvidenceEventType.TEST_OBSERVED,
            payload={
                "stage": "internal_validation",
                "independent_confirmation": False,
                "selection_used_heldout": False,
                "validation_fingerprint": validation_fingerprint,
                "metrics": {
                    name: json_safe(report.get(name))
                    for name in (
                        "best_train_nmse",
                        "best_val_nmse",
                        "best_complexity",
                        "best_val_strict_max_relative_error",
                        "best_val_relative_error_p99",
                    )
                    if name in report
                },
            },
        )
    verification = registry.verify()
    if not verification.valid:
        raise RuntimeError("evidence registry verification failed")
    return spec, spec_path, Path(evidence_registry_path)


__all__ = [
    "array_fingerprint",
    "build_hypothesis_spec",
    "persist_hypothesis_and_evidence",
]
