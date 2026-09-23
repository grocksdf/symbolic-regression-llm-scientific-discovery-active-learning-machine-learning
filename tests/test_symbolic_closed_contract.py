"""Response-free engine/adapter correctness fixtures, never efficacy evidence."""
from types import SimpleNamespace
import ast
import numpy as np
import pytest

from hypothesis_mvp.config import SymbolicConfig
from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import additive_closed_form, structural_terms
from hypothesis_mvp.symbolic.mcts_agent import MCTSSymbolicAgent
from hypothesis_mvp.symbolic.pysr_wrapper import PolynomialLassoRegressor, get_symbolic_regressor
from hypothesis_mvp.symbolic.registry import registered_engine_names
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


def test_mcts_exports_deterministic_predictively_nonredundant_pareto_bank():
    X, y = _fixture()
    cfg = SymbolicConfig(engine="mcts", expression_contract="pcpi-closed-basis-v1",
        mcts_max_iterations=48, mcts_frontier_size=4, mcts_random_seed=17)
    first, second = MCTSSymbolicAgent(cfg).fit(X, y), MCTSSymbolicAgent(cfg).fit(X, y)
    candidates = first.candidate_expressions()
    assert candidates == second.candidate_expressions()
    assert candidates[0] == first.best_expression()
    assert len(candidates) == 4 and len(set(candidates)) == len(candidates)
    assert all(structural_terms(expression, X.shape[1]) for expression in candidates)
    assert first.info()["candidate_set_method"].startswith("cross-fitted-closed-basis")
    assert first.info()["search_score_method"] == "closed-basis-amplitude-refit-crossfit-v1"
    assert first.info()["search_score_folds"] == 2
    assert first.info()["archive_size"] <= cfg.mcts_max_iterations


def test_mcts_closed_basis_search_score_is_invariant_to_external_amplitudes():
    X, y = _fixture()
    model = MCTSSymbolicAgent(SymbolicConfig(
        engine="mcts", expression_contract="pcpi-closed-basis-v1",
        mcts_score_folds=2, mcts_random_seed=17))
    model._n_features = X.shape[1]
    first_expression, first_loss, first_prediction = model._closed_basis_crossfit(
        "x0 + sin(x1)", X, y)
    second_expression, second_loss, second_prediction = model._closed_basis_crossfit(
        "7*x0 - 3*sin(x1)", X, y)
    assert structural_terms(first_expression, X.shape[1]) == structural_terms(
        second_expression, X.shape[1])
    assert first_loss == pytest.approx(second_loss, abs=1e-12)
    assert np.allclose(first_prediction, second_prediction, atol=1e-12, rtol=0.0)


def test_mcts_frontier_diversity_is_measured_against_linear_core():
    model = MCTSSymbolicAgent(SymbolicConfig(
        engine="mcts", expression_contract="pcpi-closed-basis-v1",
        mcts_frontier_size=2))
    core = np.array([1., 0., -1., 0.])
    novel = np.array([0., 1., 0., -1.])
    near_core = np.array([1., .01, -1., -.01])
    near_core /= np.linalg.norm(near_core)
    model._core_reference_signature = core / np.linalg.norm(core)
    model.best_expr = "best"
    model._archive = {
        "best": (1.0, 1.0, model._core_reference_signature),
        "near": (1.01, 1.0, near_core),
        "novel": (1.02, 1.0, novel / np.linalg.norm(novel)),
    }
    assert model._select_predictive_pareto_frontier() == ("best", "novel")
    assert "core-relative" in model.info()["candidate_set_method"]


def test_scheduler_charges_one_mcts_job_while_exporting_fixed_frontier():
    from hypothesis_mvp.symbolic import EngineScheduler
    X, y = _fixture()
    result = EngineScheduler().run(engines=("polynomial_lasso", "mcts"),
        config=SymbolicConfig(expression_contract="pcpi-closed-basis-v1",
            mcts_max_iterations=48, mcts_frontier_size=4),
        X_train=X, y_train=y, X_val=X, y_val=y, repeats=1, base_seed=17,
        max_retries=0, evaluation_budget=2, parallel=False, max_workers=1,
        timeout_s=30)
    mcts = [row for row in result.all_results if row.engine == "mcts"]
    assert result.evaluations_used == 2 and len(result.run_records) == 2
    assert 2 <= len(mcts) <= 4
    assert all(row.diagnostics["candidate_count"] == len(mcts) for row in mcts)
    assert len({row.lineage_id for row in mcts}) == len(mcts)


@pytest.mark.parametrize("engine", ["sparse_library", "additive_mechanisms"])
def test_registered_library_engines_are_deterministic_and_adapter_closed(engine):
    X, y = _fixture()
    config = SymbolicConfig(
        engine=engine, expression_contract="pcpi-closed-basis-v1")
    first = get_symbolic_regressor(config).fit(X, y)
    second = get_symbolic_regressor(config).fit(X, y)
    assert first.candidate_expressions() == second.candidate_expressions()
    assert first.best_expression() == first.candidate_expressions()[0]
    assert all(structural_terms(value, X.shape[1])
               for value in first.candidate_expressions())
    prediction = first.predict(X).reshape(-1)
    assert prediction.shape == y.shape and np.all(np.isfinite(prediction))


def test_four_skill_scheduler_preserves_one_job_per_engine_and_lineage():
    from hypothesis_mvp.symbolic import EngineScheduler
    X, y = _fixture()
    engines = registered_engine_names()
    assert engines == ("polynomial_lasso", "mcts", "sparse_library",
                       "additive_mechanisms")
    result = EngineScheduler().run(
        engines=engines,
        config=SymbolicConfig(expression_contract="pcpi-closed-basis-v1",
            mcts_max_iterations=12, mcts_frontier_size=2),
        X_train=X, y_train=y, X_val=X, y_val=y, repeats=1,
        base_seed=17, max_retries=0, evaluation_budget=len(engines),
        parallel=False, max_workers=1, timeout_s=30)
    assert not result.failures
    assert result.evaluations_used == len(engines)
    assert {row.engine for row in result.all_results} == set(engines)
    assert len({row.lineage_id for row in result.all_results}) == len(
        result.all_results)


def test_staged_engine_results_merge_without_replaying_probe_jobs():
    from hypothesis_mvp.symbolic import EngineScheduler, merge_multi_engine_results
    X, y = _fixture()
    scheduler = EngineScheduler()
    config = SymbolicConfig(
        expression_contract="pcpi-closed-basis-v1",
        mcts_max_iterations=8, mcts_frontier_size=2)
    first = scheduler.run_allocated(
        allocations={"polynomial_lasso": 1, "mcts": 1}, config=config,
        X_train=X, y_train=y, X_val=X, y_val=y, base_seed=1,
        max_retries=0, evaluation_budget=2, parallel=False,
        max_workers=1, timeout_s=30)
    second = scheduler.run_allocated(
        allocations={"mcts": 1}, config=config,
        X_train=X, y_train=y, X_val=X, y_val=y, base_seed=1000004,
        max_retries=0, evaluation_budget=1, parallel=False,
        max_workers=1, timeout_s=30)
    merged = merge_multi_engine_results(
        (first, second), evaluation_budget=3)
    assert merged.evaluations_used == 3
    assert len(merged.run_records) == 3
    assert {row.engine for row in merged.all_results} == {
        "polynomial_lasso", "mcts"}


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


@pytest.mark.parametrize("expression", [
    "0.333333333333333*(x8 + sin(x0))",
    "(x8 + sin(x0))*(-2)",
    "-2*(x8 - sin(x0))",
    "-(-2)*(x8 + sin(x0))",
])
def test_outer_scalar_sum_has_same_additive_closed_support(expression):
    assert structural_terms(expression, 9) == ("sin_x0", "x8")
    form = additive_closed_form(expression, 9)
    assert structural_terms(form, 9) == ("sin_x0", "x8")
    X, y = _fixture()
    runtime = EquationRuntime(9, refit_policy="pcpi-closed-basis-amplitudes")
    before = runtime.predict(expression, X)
    after = runtime.predict(form, X)
    np.testing.assert_allclose(before, after, rtol=1e-14, atol=1e-14)
    fitted = runtime.refit_global_constants(expression, X, y)
    assert set(structural_terms(fitted.expression, 9)) <= {"sin_x0", "x8", "intercept"}


@pytest.mark.parametrize("expression", [
    "x0*(x1 + 1)", "(x0 + 1)*x1", "2*(sin(2*x0) + x1)",
    "0.5*(x0 + x0)", "0*(x0 + x1)",
])
def test_scalar_distribution_does_not_admit_symbolic_or_invalid_structures(expression):
    with pytest.raises(ValueError):
        structural_terms(expression, 2)


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
