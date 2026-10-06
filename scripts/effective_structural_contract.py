"""Bind the new S planner to a fresh protocol and source-matched Gates."""
from scripts.effective_structural_generator import (
    LEGACY_POLICY, POLICY, policy_contract,
)


def validate_registration(protocol, gates, benchmark_source, mainline_source):
    policy = protocol.get("s_generator_policy", LEGACY_POLICY)
    if policy not in (LEGACY_POLICY, POLICY):
        raise ValueError("unknown registered S generator policy")
    if policy == LEGACY_POLICY:
        if any(gate.get("generator_policy", LEGACY_POLICY) != policy
               for gate in gates):
            raise ValueError("historical protocol cannot consume effective S Gates")
        return policy, None
    contract = policy_contract()
    if (protocol.get("s_generator_contract") != contract
            or protocol.get("historical_no_go_preserved") is not True):
        raise ValueError("effective S needs a fresh explicit contract preserving NO-GO")
    for gate in gates:
        if (gate.get("generator_policy") != policy
                or gate.get("generator_contract") != contract
                or gate.get("benchmark_source") != benchmark_source
                or gate.get("mainline_source") != mainline_source
                or gate.get("passed") is not True):
            raise ValueError("effective S requires policy/source-matched correctness Gates")
    return policy, contract


def verify_frozen_policy(freeze):
    policy = freeze.get("s_generator_policy", LEGACY_POLICY)
    if policy == POLICY:
        if (freeze.get("s_generator_contract") != policy_contract()
                or freeze.get("historical_no_go_preserved") is not True):
            raise ValueError("effective S frozen contract changed")
    elif policy != LEGACY_POLICY:
        raise ValueError("unknown frozen S generator policy")
    return policy
