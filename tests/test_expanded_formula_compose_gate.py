"""Response-free contract checks for expanded formula generation."""

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import (
    PROPOSAL_PROTOCOL_ID,
    ProposalContext,
    ProposalRuntime,
    ProtocolError,
)
from hypothesis_mvp.symbolic.registry import registered_engine_names
from hypothesis_mvp.symbolic.pysr_wrapper import get_symbolic_regressor


def _expanded_runtime() -> ProposalRuntime:
    return ProposalRuntime(
        EquationRuntime(2, refit_policy="pcpi-expanded-fixed-inner-v1"),
        2, None, candidates_per_island=2,
    )


def _context(*, quota: int = 1, gap: bool = False) -> ProposalContext:
    return ProposalContext(
        round_id=1, island="gap", parent_hash="parent",
        n_features=2, max_candidates=2, gap_directed=gap,
        new_skeleton_quota=quota,
    )


def _item(action: str = "PROPOSE_NEW_SKELETON") -> dict[str, object]:
    return {
        "candidate_id": "candidate-1",
        "parent_hash": "parent",
        "action": action,
        "equation": "exp(-0.7*x0) + log(1 + Abs(x1))",
        "rationale": "response-free grammar fixture",
    }


def test_expanded_payload_and_new_skeleton_are_admissible():
    runtime = _expanded_runtime()
    payload = runtime._proposal_payload(
        "fixture", "response-free", _context(),
        {"posterior_gap_brief": {"propose_allowed": False}}, [], [],
    )
    instruction = payload["contract"]["instruction"]
    assert payload["contract"]["expanded_formula_contract"] is True
    assert "PROPOSE_NEW_SKELETON" in instruction
    candidate, audit = runtime._candidate(
        _item(), 0, _context(), "prompt", "response",
    )
    assert candidate.action == "PROPOSE_NEW_SKELETON"
    assert audit["new_skeleton_action"] is True


def test_expanded_batch_requires_the_registered_topology_route():
    runtime = _expanded_runtime()
    parsed = {
        "protocol_id": PROPOSAL_PROTOCOL_ID,
        "round_id": 1,
        "island": "gap",
        "candidates": [{
            **_item("REPLACE"),
            "equation": "x0 + x1",
        }],
    }
    try:
        runtime._validate_batch(parsed, _context(), "prompt", "response")
    except ProtocolError as error:
        assert str(error) == "expanded_batch_missing_new_skeleton"
    else:
        raise AssertionError("expanded batch accepted without new skeleton")


def test_pysr_is_registered_for_expanded_expression_contract():
    assert "pysr" in registered_engine_names()
    # Construction may fail only when the optional Julia/PySR dependency is
    # absent; the expression contract is checked before real data access.
    config = SymbolicConfig(
        engine="pysr",
        expression_contract="pcpi-expanded-fixed-inner-v1",
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs", "tanh"],
    )
    try:
        get_symbolic_regressor(config)
    except ImportError as error:
        assert "pysr backend is unavailable" in str(error)
