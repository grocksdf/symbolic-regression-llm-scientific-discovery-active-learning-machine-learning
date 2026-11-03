from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_search_feedback import (
    mixture_location_feedback, feedback_from_posterior, rank_materials,
    attach_feedback, install_feedback,
)
from hypothesis_mvp.discovery.pcpi_adapter import freeze_expanded_formula_model, EXPANDED_FORMULA_POLICY
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def test_location_score_matches_finite_difference_of_entire_mixture():
    mu = np.array([[0., 2.], [-1., 1.], [2., -2.]])
    scale = np.array([[.5, 2.], [1., .7], [2., 1.]])
    df, weights, y = np.array([3., 8.]), np.log([.4, .6]), np.array([1., .3, 3.])
    logp, score, residual = mixture_location_feedback(mu, scale, df, weights, y)
    def shifted(delta):
        return logsumexp(t.logpdf(y[:, None], df, loc=mu + delta, scale=scale) + weights, axis=1)
    np.testing.assert_allclose(logp, shifted(0.))
    np.testing.assert_allclose(score, (shifted(1e-5) - shifted(-1e-5)) / 2e-5, rtol=1e-8)
    np.testing.assert_allclose(residual, y - mu[:, 1])
    assert not np.allclose(score, residual)


def test_gaussian_limit_and_zero_weight_component():
    logp, score, _ = mixture_location_feedback([[0., 1e6]], [[2., 1.]], [1e9, 3.], [0., -np.inf], [1.])
    assert score[0] == pytest.approx(.25, rel=1e-8)
    assert np.isfinite(logp[0])


def test_component_permutation_and_extreme_finite_values():
    arguments = (np.array([[0., 2.]]), np.array([[1., 2.]]), np.array([3., 7.]), np.log([.3, .7]))
    first = mixture_location_feedback(*arguments, [1e100])
    reordered = mixture_location_feedback(arguments[0][:, ::-1], arguments[1][:, ::-1],
                                         arguments[2][::-1], arguments[3][::-1], [1e100])
    np.testing.assert_allclose(first[0], reordered[0])
    np.testing.assert_allclose(first[1], reordered[1])
    assert np.all(np.isfinite(first[1]))


@pytest.mark.parametrize("weights,scale", [([0., 0.], [[1., 1.]]),
                                         (np.log([.5, .5]), [[0., 1.]]),
                                         (np.log([.5, .5]), [[np.nan, 1.]])])
def test_invalid_law_fails_closed(weights, scale):
    with pytest.raises(ValueError):
        mixture_location_feedback([[0., 1.]], scale, [3., 4.], weights, [0.])


def fitted():
    x = np.linspace(-1., 1., 12)[:, None]
    data = RoleDataset(DataRole.DEVELOPMENT, x, x[:, 0] ** 2 + .2 * x[:, 0])
    model = freeze_expanded_formula_model(
        [{"expression": "x0", "source": "engine:pysr"},
         {"expression": "x0**2", "source": "engine:pysr"}],
        n_features=1, prior=NormalInverseGammaPrior(), exploration_identity="a" * 64,
        coefficient_policy=EXPANDED_FORMULA_POLICY)
    engine = model.engine(model.stable_hash)
    posterior = engine.fit_batch(data.X, data.y)
    feedback = feedback_from_posterior(
        engine, posterior, data, search_row_ids=tuple(f"fit:{i}" for i in range(12)),
        forbidden_row_ids=("report:0",), bank_identity=model.stable_hash)
    return engine, posterior, data, feedback


def test_actual_posterior_density_and_alignment():
    engine, posterior, data, feedback = fitted()
    np.testing.assert_allclose(feedback.log_density, engine.predictive_logpdf(posterior, data.X, data.y))
    score = np.asarray(feedback.location_score)
    result = rank_materials(feedback, {"aligned": score, "constant": np.ones(12)}, mode="bank")
    assert result[0]["expression"] == "aligned"
    assert result[0]["priority"] > 0
    assert result[1]["constant"]
    assert attach_feedback({"allowed_edits": ["ADD"]}, feedback, mode="bank")["allowed_edits"] == ["ADD"]
    assert feedback.prompt_payload("blind")["regions"] == []


def test_sealed_role_and_overlap_rejected():
    engine, posterior, data, _ = fitted()
    for role, forbidden in ((DataRole.UNTOUCHED_HELDOUT, ("r",)),
                            (DataRole.DEVELOPMENT, ("fit:0",))):
        with pytest.raises(ValueError):
            feedback_from_posterior(engine, posterior, RoleDataset(role, data.X, data.y),
                                    search_row_ids=tuple(f"fit:{i}" for i in range(12)),
                                    forbidden_row_ids=forbidden, bank_identity="fixed")


def test_feedback_enters_real_runtime_context_without_modifying_rights():
    from hypothesis_mvp.discovery.scientific_runtime import ScientificDiscoveryRuntime
    _, _, _, feedback = fitted()
    runtime = SimpleNamespace(config=SimpleNamespace(feedback_generation_policy="legacy", new_skeleton_quota=1),
                              orchestration_context={}, evaluation=SimpleNamespace(
                                  validity_attempts=[], failure_signature=lambda *args: ()))
    current = SimpleNamespace(compact=lambda: {}, dag=SimpleNamespace(expression="x0"))
    exploration = SimpleNamespace(as_prompt_dict=lambda: {})
    before = ScientificDiscoveryRuntime._proposal_context(runtime, current, exploration, "balanced")
    install_feedback(runtime, feedback, mode="bank")
    after = ScientificDiscoveryRuntime._proposal_context(runtime, current, exploration, "balanced")
    assert before["allowed_edits"] == after["allowed_edits"]
    assert before["new_skeleton_quota"] == after["new_skeleton_quota"]
    assert after["bank_predictive_search_feedback"]["regions"]
    with pytest.raises(ValueError):
        install_feedback(runtime, feedback, mode="residual")
    with pytest.raises(ValueError):
        attach_feedback({"admission_gain": 3.}, feedback, mode="bank")


@pytest.mark.parametrize("field", ["structure_library_read", "structure_library_write",
                                  "task_local_memory_read", "task_local_memory_write"])
def test_memory_cannot_contaminate_information_condition(field):
    _, _, _, feedback = fitted()
    runtime = SimpleNamespace(config=SimpleNamespace(**{field: True}), orchestration_context={})
    with pytest.raises(ValueError, match="dedicated"):
        install_feedback(runtime, feedback, mode="bank")
