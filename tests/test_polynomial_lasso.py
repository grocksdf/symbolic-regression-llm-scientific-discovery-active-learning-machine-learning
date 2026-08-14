import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.exceptions import ConvergenceWarning

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.symbolic.pysr_wrapper import PolynomialLassoRegressor


def test_scaled_polynomial_lasso_converges_on_real_measurements() -> None:
    X, y = load_diabetes(return_X_y=True)
    X = np.asarray(X[:, :5], dtype=float).copy()
    X[:, 0] *= 1.0e6
    model = PolynomialLassoRegressor(degree=4, alpha=1.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    prediction = model.predict(X[:16]).reshape(-1)
    assert np.all(np.isfinite(prediction))
    expression = model.best_expression()
    assert "x0" in expression
    symbolic_prediction = EquationRuntime(X.shape[1]).predict(expression, X[:16])
    np.testing.assert_allclose(symbolic_prediction, prediction, rtol=1.0e-7, atol=1.0e-5)
    assert model.info()["feature_scaling"] == "standard"
    assert model.info()["iterations"] < model.info()["max_iterations"]
