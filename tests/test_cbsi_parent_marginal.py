"""Algebra/lifecycle fixtures only; no data, providers or efficacy estimates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cbsi_zero_provider_experiment.py"
SPEC = importlib.util.spec_from_file_location("cbsi_marginal_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parent_increment_is_invariant_to_inherited_bank_value():
    gain, folds = MODULE.parent_relative_gain((.3, .5), (.1, .2))
    shifted, _ = MODULE.parent_relative_gain((4.3, 6.5), (4.1, 6.2))
    assert gain == pytest.approx(.25)
    assert shifted == pytest.approx(gain)
    assert folds == pytest.approx((.2, .3))
    with pytest.raises(ValueError, match="aligned"):
        MODULE.parent_relative_gain((.3, .5), (.2,))
    with pytest.raises(ValueError, match="finite"):
        MODULE.parent_relative_gain((.3, np.nan), (.1, .2))


@pytest.mark.parametrize("gain,increment,depth,expected", [
    (.2, -.1, 0, "no_gain_over_parent"),
    (.2, 0., 0, "no_gain_over_parent"),
    (.2, None, 0, "parent_increment_unavailable"),
    (-.2, .1, 0, "no_gain_over_initial_bank"),
    (.2, .1, 3, "depth_budget_exhausted"),
    (.2, .1, 0, "promoted"),
])
def test_frontier_promotion_requires_increment_not_just_parent_inheritance(
        gain, increment, depth, expected):
    assert MODULE.frontier_promotion_reason(gain, increment, depth, 3) == expected


def test_historical_freeze_fails_before_any_data_or_output_access(tmp_path):
    freeze_path = tmp_path / "historical.json"
    freeze_path.write_text(json.dumps({
        "execution_authorized": True,
        "protocol_id": "cbsi-contextual-frontier-v3",
        "hdf5": "must-never-be-opened.hdf5",
    }))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="historical freezes"):
        MODULE.main(["--freeze", str(freeze_path), "--output-root", str(output)])
    assert not output.exists()


def test_terminal_frontier_parent_cannot_create_over_budget_descendant():
    base = MODULE.FrontierParent("x0", "base", "base", 0)
    terminal = MODULE.FrontierParent("x0*x1", "terminal", "terminal", 3)
    middle = MODULE.FrontierParent("x0+x1", "middle", "middle", 2)
    frontier = [{"parent": terminal, "search_gain": 10.},
                {"parent": middle, "search_gain": .1}]
    assert MODULE.select_frontier_parent(frontier, 1, base, max_depth=3) == (middle, True)
    assert MODULE.select_frontier_parent(frontier[:1], 1, base, max_depth=3) == (base, False)


def test_runner_uses_marginal_reward_and_seals_selection_before_evaluation(monkeypatch):
    from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
        ZERO_PROVIDER_OPERATORS,
    )

    events = []
    class Scorer:
        def __init__(self, *args):
            self.score_calls = 0
            self.sealed = False
        def crossfit_gain(self, expression):
            assert not self.sealed
            self.score_calls += 1
            events.append("search")
            return (.5, .5) if expression == "x0 + x1" else (.2, .2)
        def freeze_selection(self, expression):
            assert not self.sealed
            self.sealed = True
            self.selected = expression
            events.append("sealed")
        def evaluation_gain(self, expression):
            assert self.sealed and expression == self.selected
            events.append("evaluation")
            return -.1, [.5, .5], {"converged": True}
    monkeypatch.setattr(MODULE, "CrossfitBankScorer", Scorer)
    x = np.column_stack((np.linspace(.2, 1.1, 20), np.linspace(.7, 1.8, 20)))
    parent = SimpleNamespace(expression="x0 + x1", structural_hash="p", lineage_id="p")
    freeze = {"operators": ZERO_PROVIDER_OPERATORS, "budget": {
        "edit_evaluations_per_arm": 8, "contextual_credit_weight": .5,
        "complexity_cost_weight": 0., "frontier_max_depth": 3,
        "frontier_size": 8, "frontier_min_search_gain": 0.,
    }}
    opportunities = tuple({
        "parent": parent, "donor": "sin(x0)", "site_path": (),
        "opportunity_seed": 101 + i, "opportunity_id": f"o{i}",
    } for i in range(4))
    result = MODULE.run_arm(
        "cbsi_frontier", freeze, {"seed": 2026, "family": "fixture", "task": "fixture"},
        SimpleNamespace(candidates=[parent]), [{"expression": "x0"}],
        SimpleNamespace(development=SimpleNamespace(X=x)), None, None, None,
        {}, (x,), opportunities)
    observations = [r for r in result["policy"]["observations"] if r["raw_gain"] is not None]
    assert observations
    assert all(r["raw_gain"] == pytest.approx(-.3) for r in observations)
    assert result["frontier_parent_reuses"] == 0
    assert result["final_frontier_size"] == 0
    assert result["independent_development_evaluation_gain"] == -.1
    assert events[-2:] == ["sealed", "evaluation"]
    assert result["implementation_contract"] == MODULE.IMPLEMENTATION_CONTRACT
