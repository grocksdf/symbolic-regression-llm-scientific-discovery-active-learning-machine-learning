"""No-observation VED confirmation source registration tests."""

import json
from pathlib import Path

import pytest

from hypothesis_mvp.discovery.ved_confirmation_source import (
    validate_ved_source_registration,
)


def _registration():
    path = (Path(__file__).resolve().parents[1] / "configs" /
            "scientific_ved_confirmation_source_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_ved_source_registration_is_closed_and_grammar_compatible():
    registration = validate_ved_source_registration(_registration())
    assert registration["execution_authorized"] is False
    assert registration["grammar_contract"] == "pcpi-closed-basis-v1"
    assert registration["target"] not in registration["features"]


def test_ved_source_registration_rejects_open_confirmation_identity():
    registration = _registration()
    registration["open_members"]["acquisition_pool"] = registration[
        "open_members"]["development"]
    with pytest.raises(ValueError, match="roles"):
        validate_ved_source_registration(registration)
