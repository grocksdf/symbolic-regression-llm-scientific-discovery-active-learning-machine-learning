import copy

import pytest

from scripts.effective_structural_contract import (
    validate_registration, verify_frozen_policy,
)
from scripts.effective_structural_generator import LEGACY_POLICY, POLICY, policy_contract
from scripts.run_matched_attribution_matrix import _s_command


def inputs():
    protocol = {"s_generator_policy": POLICY,
                "s_generator_contract": policy_contract(),
                "historical_no_go_preserved": True}
    benchmark, mainline = {"commit": "b"}, {"commit": "m"}
    gate = {"passed": True, "generator_policy": POLICY,
            "generator_contract": policy_contract(),
            "benchmark_source": benchmark, "mainline_source": mainline}
    return protocol, [copy.deepcopy(gate) for _ in range(3)], benchmark, mainline


def test_new_policy_requires_explicit_fresh_registration_and_same_source_gates():
    protocol, gates, benchmark, mainline = inputs()
    assert validate_registration(protocol, gates, benchmark, mainline) == (POLICY, policy_contract())
    assert verify_frozen_policy(protocol) == POLICY


@pytest.mark.parametrize("field,value", [
    ("generator_policy", LEGACY_POLICY), ("generator_contract", {}),
    ("benchmark_source", {"commit": "other"}),
    ("mainline_source", {"commit": "other"}), ("passed", False)])
def test_effective_policy_rejects_crossed_gate_identity(field, value):
    protocol, gates, benchmark, mainline = inputs()
    gates[1][field] = value
    with pytest.raises(ValueError, match="policy/source-matched"):
        validate_registration(protocol, gates, benchmark, mainline)


def test_historical_freeze_cannot_silently_use_new_comparator():
    assert verify_frozen_policy({}) == LEGACY_POLICY
    _, gates, benchmark, mainline = inputs()
    with pytest.raises(ValueError, match="historical protocol"):
        validate_registration({}, gates, benchmark, mainline)
    assert validate_registration({}, [{"passed": True}], benchmark, mainline) == (LEGACY_POLICY, None)


@pytest.mark.parametrize("field", ["s_generator_contract", "historical_no_go_preserved"])
def test_effective_contract_cannot_be_omitted(field):
    protocol, gates, benchmark, mainline = inputs()
    del protocol[field]
    with pytest.raises(ValueError):
        validate_registration(protocol, gates, benchmark, mainline)
    with pytest.raises(ValueError):
        verify_frozen_policy(protocol)


def test_matrix_forwards_only_frozen_policy_to_S_child():
    protocol, _, _, _ = inputs()
    freeze = {**protocol, "hdf5": {"path": "/fixture.h5"}}
    item = {"root": "/fixture/run", "family": "phys_osc", "task": "T", "seed": 1}
    command = _s_command(freeze, item)
    assert command[command.index("--generator-policy") + 1] == POLICY
    command = _s_command({"hdf5": {"path": "/fixture.h5"}}, item)
    assert command[command.index("--generator-policy") + 1] == LEGACY_POLICY


def test_unknown_policy_is_not_a_fallback_request():
    with pytest.raises(ValueError):
        verify_frozen_policy({"s_generator_policy": "unknown"})
