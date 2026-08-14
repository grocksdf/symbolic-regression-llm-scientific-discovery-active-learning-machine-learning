from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pytest

import hypothesis_mvp.discovery as discovery
from hypothesis_mvp.data import DataRole, RoleDataset, SelectionData
from hypothesis_mvp.discovery import DiscoveryConfig, discover_from_selection
from hypothesis_mvp.hypotheses import (
    EvidenceEventType,
    EvidenceRegistry,
    HypothesisSpec,
    VariableSpec,
)


def _selection() -> SelectionData:
    development = RoleDataset(
        DataRole.DEVELOPMENT,
        np.arange(24, dtype=float).reshape(12, 2),
        np.arange(12, dtype=float),
    )
    validation = RoleDataset(
        DataRole.VALIDATION,
        np.arange(24, 40, dtype=float).reshape(8, 2),
        np.arange(8, dtype=float) + 20.0,
    )
    return SelectionData(
        development,
        validation,
        None,
        (
            (development.role.value, development.fingerprint),
            (validation.role.value, validation.fingerprint),
        ),
    )


def test_public_surface_is_frozen_to_selection_and_confirmation_protocols() -> None:
    expected = {
        "ConfirmationResult",
        "DiscoveryAgent",
        "DiscoveryAgentConfig",
        "DiscoveryAgentResult",
        "DiscoveryConfig",
        "DiscoveryPhase",
        "DiscoveryRunResult",
        "DiscoveryState",
        "ProviderSettings",
        "RuntimeEvent",
        "confirm_frozen_hypothesis",
        "discover_from_selection",
    }
    assert set(discovery.__all__) == expected
    for removed in (
        "ConfirmationEvidence",
        "DiscoveryStores",
        "DiscoveryTaskContext",
        "discover_from_arrays",
        "promote_staged_knowledge",
        "record_confirmation_evidence",
    ):
        assert not hasattr(discovery, removed)


def test_selection_entrypoint_has_no_hidden_data_capability() -> None:
    signature = inspect.signature(discover_from_selection)
    parameters = tuple(signature.parameters.values())
    assert parameters[0].name == "selection"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters[1].kind is inspect.Parameter.VAR_KEYWORD
    assert "heldout" not in SelectionData.__dataclass_fields__


@dataclass(frozen=True)
class _SelectionWithHeldout:
    development: RoleDataset
    validation: RoleDataset
    heldout: RoleDataset


def test_selection_entrypoint_rejects_a_surface_with_heldout_field() -> None:
    selection = _selection()
    heldout = RoleDataset(
        DataRole.UNTOUCHED_HELDOUT,
        np.ones((4, 2)),
        np.ones(4),
    )
    with pytest.raises(TypeError, match="held-out capability"):
        discover_from_selection(
            _SelectionWithHeldout(selection.development, selection.validation, heldout),
            task_name="leakage_contract",
            knowledge_dir="unused",
        )


def test_discovery_config_keeps_only_supported_profile_controls() -> None:
    config = DiscoveryConfig.from_mapping({
        "evaluation_budget": 1000,
        "candidates_per_island": 4,
        "structure_library_max_entries": 20,
    })
    assert config.evaluation_budget == 1000
    assert config.candidates_per_island == 4
    assert config.structure_library_max_entries == 20
    assert config.islands
    assert not hasattr(config, "profile_name")
    assert not hasattr(config, "exemplars_per_prompt")


def test_confirmation_requires_untouched_heldout_role(tmp_path) -> None:
    variable = VariableSpec(name="x0", unit="1", description="input")
    target = VariableSpec(name="y", unit="1", description="response")
    hypothesis = HypothesisSpec.create(
        hypothesis_id="hyp-confirmation-role",
        expression="x0",
        variables=(variable,),
        target=target,
        domain="test",
        assumptions=("finite measurements",),
        mechanism="identity response",
        falsifiers=("independent NMSE fails",),
        provenance={"test": True},
    )
    validation = RoleDataset(
        DataRole.VALIDATION,
        np.ones((2, 1)),
        np.ones(2),
    )
    with pytest.raises(ValueError, match="untouched-heldout role"):
        discovery.confirm_frozen_hypothesis(
            hypothesis=hypothesis,
            untouched_heldout=validation,
            evidence_registry_path=tmp_path / "evidence.jsonl",
            maximum_nmse=0.1,
        )


def test_confirmation_is_single_use_and_records_verified_evidence(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.jsonl"
    variable = VariableSpec(name="x0", unit="1", description="input")
    target = VariableSpec(name="y", unit="1", description="response")
    hypothesis = HypothesisSpec.create(
        hypothesis_id="hyp-confirmation-contract",
        expression="x0",
        variables=(variable,),
        target=target,
        domain="test",
        assumptions=("finite measurements",),
        mechanism="identity response",
        falsifiers=("independent NMSE fails",),
        provenance={"test": True},
    )
    registry = EvidenceRegistry(evidence_path)
    registry.append(
        hypothesis_id=hypothesis.hypothesis_id,
        event_type=EvidenceEventType.PROPOSED,
        payload={"expression": hypothesis.expression},
    )
    heldout = RoleDataset(
        DataRole.UNTOUCHED_HELDOUT,
        np.arange(4, dtype=float).reshape(4, 1),
        np.arange(4, dtype=float),
    )
    result = discovery.confirm_frozen_hypothesis(
        hypothesis=hypothesis,
        untouched_heldout=heldout,
        evidence_registry_path=evidence_path,
        maximum_nmse=0.1,
    )
    assert result.passed is True
    assert result.metrics["nmse"] == pytest.approx(0.0)
    events = registry.events(hypothesis_id=hypothesis.hypothesis_id)
    assert [event.event_type for event in events] == [
        EvidenceEventType.PROPOSED,
        EvidenceEventType.TEST_OBSERVED,
        EvidenceEventType.STATUS_CHANGED,
    ]
    with pytest.raises(RuntimeError, match="already exists"):
        discovery.confirm_frozen_hypothesis(
            hypothesis=hypothesis,
            untouched_heldout=heldout,
            evidence_registry_path=evidence_path,
            maximum_nmse=0.1,
        )
