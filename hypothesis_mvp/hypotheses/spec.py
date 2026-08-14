"""Typed, serializable contract for machine-generated scientific hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .ast_validator import ASTValidationResult, SafeExpressionValidator


HYPOTHESIS_SPEC_SCHEMA_VERSION = "hypothesis-spec-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTABLE = "testable"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    DORMANT = "dormant"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid variable name: {self.name!r}")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(f"variable {self.name!r} requires an explicit unit")
        if not isinstance(self.description, str):
            raise TypeError("variable description must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariableSpec":
        allowed = {"name", "unit", "description"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown VariableSpec fields: {sorted(unknown)}")
        return cls(
            name=str(value["name"]),
            unit=str(value["unit"]),
            description=str(value.get("description", "")),
        )


def _string_tuple(name: str, values: Sequence[str], *, nonempty: bool) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    output = tuple(str(value).strip() for value in values)
    if any(not value for value in output):
        raise ValueError(f"{name} cannot contain empty strings")
    if nonempty and not output:
        raise ValueError(f"{name} must contain at least one item")
    return output


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain JSON-safe finite values") from error
    return json.loads(encoded)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    expression: str
    variables: tuple[VariableSpec, ...]
    target: VariableSpec
    domain: str
    assumptions: tuple[str, ...]
    mechanism: str
    falsifiers: tuple[str, ...]
    provenance: Mapping[str, Any]
    created_at_utc: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: str | None = None
    edit_operator: str | None = None
    schema_version: str = HYPOTHESIS_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HYPOTHESIS_SPEC_SCHEMA_VERSION:
            raise ValueError(f"unsupported HypothesisSpec schema: {self.schema_version}")
        if not _IDENTIFIER.fullmatch(self.hypothesis_id):
            raise ValueError(f"invalid hypothesis_id: {self.hypothesis_id!r}")
        if self.parent_id is not None and not _IDENTIFIER.fullmatch(self.parent_id):
            raise ValueError(f"invalid parent_id: {self.parent_id!r}")
        if (self.parent_id is None) != (self.edit_operator is None):
            raise ValueError("parent_id and edit_operator must be supplied together")
        if self.parent_id == self.hypothesis_id:
            raise ValueError("a hypothesis cannot be its own parent")
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("expression must be a non-empty RHS expression")
        variables = tuple(self.variables)
        if not variables:
            raise ValueError("at least one input variable is required")
        names = [value.name for value in variables]
        if len(names) != len(set(names)):
            raise ValueError("input variable names must be unique")
        if self.target.name in names:
            raise ValueError("target name cannot also be an input variable")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be non-empty")
        if not isinstance(self.mechanism, str) or not self.mechanism.strip():
            raise ValueError("mechanism must be non-empty")
        assumptions = _string_tuple("assumptions", self.assumptions, nonempty=True)
        falsifiers = _string_tuple("falsifiers", self.falsifiers, nonempty=True)
        try:
            timestamp = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("created_at_utc must be an ISO-8601 timestamp") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("created_at_utc must include a timezone")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "falsifiers", falsifiers)
        object.__setattr__(
            self,
            "provenance",
            _freeze_json(_json_copy(self.provenance)),
        )
        if not isinstance(self.status, HypothesisStatus):
            object.__setattr__(self, "status", HypothesisStatus(str(self.status)))

    @classmethod
    def create(
        cls,
        *,
        hypothesis_id: str,
        expression: str,
        variables: Sequence[VariableSpec],
        target: VariableSpec,
        domain: str,
        assumptions: Sequence[str],
        mechanism: str,
        falsifiers: Sequence[str],
        provenance: Mapping[str, Any],
        status: HypothesisStatus = HypothesisStatus.PROPOSED,
        parent_id: str | None = None,
        edit_operator: str | None = None,
    ) -> "HypothesisSpec":
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return cls(
            hypothesis_id=hypothesis_id,
            expression=expression,
            variables=tuple(variables),
            target=target,
            domain=domain,
            assumptions=tuple(assumptions),
            mechanism=mechanism,
            falsifiers=tuple(falsifiers),
            provenance=provenance,
            created_at_utc=created,
            status=status,
            parent_id=parent_id,
            edit_operator=edit_operator,
        )

    def validate_expression(
        self,
        validator: SafeExpressionValidator | None = None,
    ) -> ASTValidationResult:
        engine = validator or SafeExpressionValidator()
        return engine.validate(
            self.expression,
            allowed_variables={variable.name for variable in self.variables},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "expression": self.expression,
            "variables": [value.to_dict() for value in self.variables],
            "target": self.target.to_dict(),
            "domain": self.domain,
            "assumptions": list(self.assumptions),
            "mechanism": self.mechanism,
            "falsifiers": list(self.falsifiers),
            "provenance": _thaw_json(self.provenance),
            "created_at_utc": self.created_at_utc,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "edit_operator": self.edit_operator,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def content_sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HypothesisSpec":
        allowed = {
            "schema_version",
            "hypothesis_id",
            "expression",
            "variables",
            "target",
            "domain",
            "assumptions",
            "mechanism",
            "falsifiers",
            "provenance",
            "created_at_utc",
            "status",
            "parent_id",
            "edit_operator",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown HypothesisSpec fields: {sorted(unknown)}")
        required = allowed - {"parent_id", "edit_operator", "status", "schema_version"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing HypothesisSpec fields: {sorted(missing)}")
        raw_variables = value["variables"]
        if not isinstance(raw_variables, Sequence) or isinstance(raw_variables, (str, bytes)):
            raise TypeError("variables must be a sequence")
        raw_target = value["target"]
        if not isinstance(raw_target, Mapping):
            raise TypeError("target must be an object")
        provenance = value["provenance"]
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be an object")
        return cls(
            schema_version=str(
                value.get("schema_version", HYPOTHESIS_SPEC_SCHEMA_VERSION)
            ),
            hypothesis_id=str(value["hypothesis_id"]),
            expression=str(value["expression"]),
            variables=tuple(VariableSpec.from_dict(item) for item in raw_variables),
            target=VariableSpec.from_dict(raw_target),
            domain=str(value["domain"]),
            assumptions=tuple(value["assumptions"]),
            mechanism=str(value["mechanism"]),
            falsifiers=tuple(value["falsifiers"]),
            provenance=provenance,
            created_at_utc=str(value["created_at_utc"]),
            status=HypothesisStatus(str(value.get("status", "proposed"))),
            parent_id=(
                None if value.get("parent_id") is None else str(value["parent_id"])
            ),
            edit_operator=(
                None
                if value.get("edit_operator") is None
                else str(value["edit_operator"])
            ),
        )
"""Append-only, hash-chained evidence events for hypothesis discovery."""


