"""Response-free engine/adapter correctness fixtures, never efficacy evidence."""
from types import SimpleNamespace
import ast
import numpy as np
import pytest

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import structural_terms
from hypothesis_mvp.symbolic.mcts_agent import MCTSSymbolicAgent
from hypothesis_mvp.symbolic.pysr_wrapper import PolynomialLassoRegressor, get_symbolic_regressor
from hypothesis_mvp.symbolic.typed_grammar import Var, Unary, expand_ast


def _fixture():
    X = np.random.default_rng(7).uniform(-1, 1, (32, 9))
    return X, 1 + .4*X[:, 0] - .7*X[:, 1]**2


def test_polynomial_library_is_bounded_before_one_fit_and_predictor_is_not_trimmed():
    X, y = _fixture()
    model = PolynomialLassoRegressor(expression_contract="pcpi-closed-basis-v1").fit(X, y)
    equation = EquationRuntime(9, refit_policy="pcpi-closed-basis-amplitudes")
    expression = equation.dag(model.best_expression()).expression
    support = structural_terms(expression, 9)
    assert len(list(ast.walk(ast.parse(expression)))) <= 257
    assert len(model._selected_features) < len(model._poly.powers_)
    assert len(model._model.coef_) == len(model._selected_features)
    assert len(support) == np.count_nonzero(model._model.coef_) + 1
    np.testing.assert_allclose(equation.predict(expression, X), model.predict(X).ravel(), atol=1e-8)
    fitted = equation.refit_global_constants(expression, X, y)
    assert set(structural_terms(fitted.expression, 9)) <= set(support) | {"intercept"}
    assert model.info()["lasso_fit_count"] == 1
    assert model.info()["library_selection"] == "train-correlation-pre-fit-ast-cap"


def test_library_selection_is_deterministic_and_receives_training_only():
    X, y = _fixture()
    a = PolynomialLassoRegressor(expression_contract="pcpi-closed-basis-v1").fit(X, y)
    b = PolynomialLassoRegressor(expression_contract="pcpi-closed-basis-v1").fit(X, y)
    np.testing.assert_array_equal(a._selected_features, b._selected_features)
    assert a.best_expression() == b.best_expression()


def test_mcts_restricts_generation_and_consumes_adapter_supported_output():
    X, y = _fixture()
    cfg = SymbolicConfig(engine="mcts", expression_contract="pcpi-closed-basis-v1",
        mcts_max_iterations=12, seed_expressions=["exp(x0)", "sin(x0**2)", "x0/x1"])
    model = MCTSSymbolicAgent(cfg).fit(X, y)
    support = structural_terms(model.best_expression(), 9)
    assert support and model.info()["contract_rejections"] > 0
    assert not model._contract_admits("sin(2*x0)")
    assert not model._contract_admits("x0**5")
    assert not model._contract_admits("x0 + x0")
    equation = EquationRuntime(9, refit_policy="pcpi-closed-basis-amplitudes")
    result = equation.refit_global_constants(model.best_expression(), X, y)
    assert set(structural_terms(result.expression, 9)) <= set(support) | {"intercept"}


def test_invalid_unary_expansions_do_not_exhaust_valid_generation_slots():
    def admit(expression):
        try: structural_terms(expression, 1); return True
        except ValueError: return False
    candidates = expand_ast(Unary("sin", Var(0)), n_features=1,
        allowed_unary=["exp", "log", "sin", "cos"], allowed_binary=["+"],
        constants=[1.], rng=np.random.default_rng(0), max_depth=6, max_nodes=40,
        max_new=2, candidate_validator=admit)
    assert candidates and all(admit(c.to_string()) for c in candidates)
    assert all("exp" not in c.to_string() and "log" not in c.to_string() for c in candidates)


@pytest.mark.parametrize("policy,contract", [("global-constants", "unrestricted"),
    ("pcpi-closed-basis-amplitudes", "pcpi-closed-basis-v1")])
def test_production_agent_wires_same_contract_and_registered_budget(monkeypatch, policy, contract):
    agent = DiscoveryAgent(DiscoveryAgentConfig(refit_policy=policy, engine_budget=2,
        engine_repeats=1, search_iterations=24, engine_workers=1))
    seen = {}
    def run(**kwargs): seen.update(kwargs); return object()
    monkeypatch.setattr(agent.scheduler, "run", run)
    X, y = _fixture()
    role = SimpleNamespace(X=X, y=y)
    agent._run_engines(SimpleNamespace(development=role, validation=role), 0)
    assert seen["config"].expression_contract == contract
    assert seen["config"].mcts_max_iterations == 24
    assert seen["evaluation_budget"] == 2
    assert seen["engines"] == ("polynomial_lasso", "mcts")


def test_unknown_or_unimplemented_contract_does_not_fall_back():
    with pytest.raises(ValueError, match="unknown"):
        get_symbolic_regressor(SymbolicConfig(expression_contract="unknown"))
    with pytest.raises(ValueError, match="does not implement"):
        get_symbolic_regressor(SymbolicConfig(engine="pysr", expression_contract="pcpi-closed-basis-v1"))
    with pytest.raises(ValueError, match="degree"):
        PolynomialLassoRegressor(degree=5, expression_contract="pcpi-closed-basis-v1")


def test_both_production_engine_outputs_survive_initial_discovery_validation(tmp_path):
    from hypothesis_mvp.discovery.factory import build_scientific_discovery_runtime
    from hypothesis_mvp.discovery.initializer import normalize_candidates
    from hypothesis_mvp.symbolic import EngineScheduler
    X, y = _fixture()
    engines = EngineScheduler().run(engines=("polynomial_lasso", "mcts"),
        config=SymbolicConfig(expression_contract="pcpi-closed-basis-v1", mcts_max_iterations=12),
        X_train=X, y_train=y, X_val=X, y_val=y, repeats=1, base_seed=7,
        max_retries=0, evaluation_budget=2, parallel=False, max_workers=1, timeout_s=30)
    assert not engines.failures and engines.evaluations_used == 2
    seeds, rejected = normalize_candidates([{"expression": row.expression,
        "source": "engine:"+row.engine, "lineage_id": row.lineage_id}
        for row in engines.all_results], n_features=9, X_probe=X)
    assert rejected == 0 and len(seeds) == 2
    runtime = build_scientific_discovery_runtime(n_features=9,
        config={"refit_policy": "pcpi-closed-basis-amplitudes", "evaluation_budget": 48,
                "llm_evaluation_reserve": 16},
        library_path=tmp_path / "library.jsonl", ledger_path=tmp_path / "ledger.jsonl")
    _, report = runtime.run(X_train=X, y_train=y, X_val=X, y_val=y,
                           base_candidates=seeds, refinement_enabled=False)
    assert {row["source"] for row in report["evaluated_hypothesis_bank"]} == {
        "engine:polynomial_lasso", "engine:mcts"}, str(report["rejected_candidates"])
    assert runtime.evaluation.budget.used <= 32


@pytest.mark.parametrize("function", ["sin", "cos", "tanh"])
@pytest.mark.parametrize("expression", ["-2*{f}(x0)", "{f}(x0)*(-2)", "(-2)*{f}(x0)"])
def test_signed_external_nonlinear_amplitudes_have_the_same_support(function, expression):
    assert structural_terms(expression.format(f=function), 1) == (function+"_x0",)
    with pytest.raises(ValueError):
        structural_terms(function+"(-2*x0)", 1)
