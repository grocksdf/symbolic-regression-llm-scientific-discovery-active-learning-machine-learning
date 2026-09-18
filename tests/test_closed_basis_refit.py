"""Algebraic refit correctness fixtures, never experimental evidence."""
import numpy as np
import pytest

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime


def test_nonlinear_arguments_and_monomials_survive_refit():
    X = np.linspace(-2, 2, 32)[:, None]
    y = 1.2 + 2.3 * np.tanh(X[:, 0]) - .4 * X[:, 0] ** 2
    runtime = EquationRuntime(1, refit_policy="pcpi-closed-basis-amplitudes")
    result = runtime.refit_global_constants("3*tanh(x0) + 2*x0**2 + 7", X, y)
    assert set(structural_terms(result.expression, 1)) == {"intercept", "tanh_x0", "x0_sq"}
    assert np.max(abs(runtime.predict(result.expression, X) - y)) < 1e-5


@pytest.mark.parametrize("expression", ["tanh(.01*x0)", "sin(x0**3)", "x0/(1+x0**2)"])
def test_incompatible_structure_is_not_silently_projected(expression):
    X = np.linspace(0, 1, 16)[:, None]
    runtime = EquationRuntime(1, refit_policy="pcpi-closed-basis-amplitudes")
    with pytest.raises(ValueError):
        runtime.refit_global_constants(expression, X, X[:, 0])


def test_factory_binds_policy_to_all_evaluation_refits(tmp_path):
    runtime = build_scientific_discovery_runtime(n_features=1,
        config=DiscoveryConfig.from_mapping({"refit_policy": "pcpi-closed-basis-amplitudes"}),
        library_path=tmp_path / "library.jsonl", ledger_path=tmp_path / "ledger.jsonl")
    assert runtime.evaluation.runtime.refit_policy == "pcpi-closed-basis-amplitudes"
    with pytest.raises(ValueError):
        EquationRuntime(1, refit_policy="silent-fallback")
