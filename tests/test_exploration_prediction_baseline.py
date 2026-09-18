"""Inference correctness diagnostic fixtures, not real efficacy evidence."""
import numpy as np
import pytest

from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.exploration_runtime import ExplorationRuntime


def _runtime():
    equation = EquationRuntime(1)
    return equation, ExplorationRuntime(equation, DiscoveryConfig.from_mapping({
        "exploration_max_depth": 1, "exploration_max_primitives": 16}))


@pytest.mark.parametrize("offset", [0., 3., -7.])
@pytest.mark.parametrize("variation", [0., 1e-14])
def test_constant_and_near_constant_baseline_is_not_filtered(offset, variation):
    equation, runtime = _runtime()
    X = np.linspace(-1, 1, 16)[:, None]
    prediction = offset + variation * X[:, 0]
    terms, _ = runtime._build_grammar(X, 2 + X[:, 0], prediction, equation.dag(str(offset)))
    baseline = [term for term in terms if term.expression == "y_hat"]
    assert len(baseline) == 1
    assert np.array_equal(baseline[0].values, prediction)
    selected, identity, _ = runtime._select_terms(terms, 2 + X[:, 0])
    assert all(np.isfinite(value) for value in identity.values())
    assert any(terms[index].expression == "y_hat" for index in selected)
    assert any(terms[index].expression == "x0" for index in selected)


def test_constant_prediction_complete_solve_remains_finite():
    from types import SimpleNamespace
    equation, runtime = _runtime()
    X = np.linspace(-1, 1, 16)[:, None]
    result = runtime.solve(X, 2 + X[:, 0], np.zeros(16), equation.dag("0"),
                           SimpleNamespace(val_nmse=1., val_p99=1., val_strict=1.,
                                           as_dict=lambda: {}))
    assert np.isfinite(result.cv_objective)
    assert np.isfinite(result.identity_cv_objective)
    assert any(row["expression"] == "x0" for row in result.selected_terms)


@pytest.mark.parametrize("prediction", [[], [1.], [np.nan]*16, [np.inf]*16])
def test_invalid_baseline_fails_explicitly(prediction):
    equation, runtime = _runtime()
    with pytest.raises(ValueError, match="finite aligned current predictions"):
        runtime._base_terms(np.ones((16, 1)), np.asarray(prediction), equation.dag("1"))


def test_optional_constant_terms_still_filtered_and_aliases_deduplicated():
    equation, runtime = _runtime()
    X = np.linspace(-1, 1, 16)[:, None]
    terms = runtime._base_terms(X, X[:, 0], equation.dag("x0"))
    assert [term.expression for term in terms] == ["y_hat"]
    assert runtime._safe_values(np.ones(16)) is None


def test_mandatory_prediction_values_are_not_clipped():
    equation, runtime = _runtime()
    X = np.linspace(-1, 1, 128)[:, None]
    prediction = np.zeros(128)
    prediction[-1] = 1e8
    terms = runtime._base_terms(X, prediction, equation.dag("x0"))
    assert np.array_equal(terms[0].values, prediction)


def test_all_constant_inputs_have_a_valid_intercept_only_baseline():
    equation, runtime = _runtime()
    y = np.linspace(1, 2, 16)
    terms, _ = runtime._build_grammar(np.ones((16, 1)), y, np.ones(16), equation.dag("1"))
    selected, identity, _ = runtime._select_terms(terms, y)
    assert [terms[index].expression for index in selected] == ["y_hat"]
    assert all(np.isfinite(value) for value in identity.values())
