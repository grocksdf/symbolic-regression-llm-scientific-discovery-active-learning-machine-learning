"""P3H.4 likelihood-power-specific prequential residual state boundaries."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from hypothesis_mvp.pcpi import (
    P3H4_OBSERVATION_ORDER,
    P3H4_RESIDUAL_FAMILY_METHOD,
    P3H4_STATE_BINDING,
    PosteriorModel,
    advance_likelihood_power_residual_family,
    reconstruct_conditioned_likelihood_power_residual_family,
    reconstruct_likelihood_power_residual_family,
)
from hypothesis_mvp.pcpi.real_acquisition import _bound_semiparametric_residual_laws
from hypothesis_mvp.pcpi.reference import (
    DyadicPolyaTreeResidualModel,
    SequentialReferencePosterior,
    fit_bank_preconditioner,
    generic_real_bank,
    predictive_cdf,
)
import hypothesis_mvp.pcpi.likelihood_power_residuals as implementation
from scripts import run_pcpi_p3h4_likelihood_power_residual_family_correctness as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p3h_4_likelihood_power_residual_family_correctness.json"


def _history() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-1.4, 1.4, 12)
    actions = np.column_stack((grid, np.cos(0.7 * grid)))
    targets = 0.25 + 0.6 * grid - 0.18 * np.square(grid) + 0.07 * np.sin(grid)
    return actions, targets


def _engines() -> tuple[SequentialReferencePosterior, ...]:
    actions, _ = _history()
    bank = generic_real_bank(actions.shape[1])
    preconditioner = fit_bank_preconditioner(bank, actions)
    return tuple(
        SequentialReferencePosterior(bank, power, preconditioner)
        for power in (1.0, 0.25, 0.5)
    )


def test_family_is_power_sorted_and_each_target_builds_its_own_raw_pits() -> None:
    actions, targets = _history()
    family = reconstruct_likelihood_power_residual_family(
        _engines(), actions, targets
    )
    assert family.likelihood_powers == (0.25, 0.5, 1.0)
    assert family.method == P3H4_RESIDUAL_FAMILY_METHOD
    assert family.observation_order == P3H4_OBSERVATION_ORDER
    assert family.state_binding == P3H4_STATE_BINDING
    assert family.observation_count == len(targets)
    assert len({state.target_hash for state in family.model_states}) == 3
    pit_sequences = tuple(state.residual_state.raw_pits for state in family.model_states)
    assert len(set(pit_sequences)) == 3
    assert all(len(values) == len(targets) for values in pit_sequences)


def test_each_raw_pit_uses_only_its_own_power_and_strict_prefix() -> None:
    actions, targets = _history()
    family = reconstruct_likelihood_power_residual_family(
        _engines(), actions, targets
    )
    model = DyadicPolyaTreeResidualModel()
    for state in family.model_states:
        manual = model.prior_state()
        for index, target in enumerate(targets):
            posterior = (
                state.engine.prior_posterior()
                if index == 0
                else state.engine.fit_batch(actions[:index], targets[:index])
            )
            pit = predictive_cdf(
                state.engine,
                posterior,
                actions[index : index + 1],
                np.asarray([target]),
                clip=None,
            )
            manual = model.update(manual, float(pit[0]))
        assert manual.raw_pits == state.residual_state.raw_pits
        expected = state.engine.fit_batch(actions, targets)
        np.testing.assert_allclose(
            [item.probability for item in state.posterior.members],
            [item.probability for item in expected.members],
            atol=0.0,
            rtol=0.0,
        )


def test_chronological_order_changes_residual_state_but_not_batch_posterior() -> None:
    actions, targets = _history()
    engines = _engines()
    forward = reconstruct_likelihood_power_residual_family(engines, actions, targets)
    reverse = reconstruct_likelihood_power_residual_family(
        engines, actions[::-1], targets[::-1]
    )
    assert forward.history_commitment != reverse.history_commitment
    assert forward.stable_hash != reverse.stable_hash
    assert any(
        left.residual_state.raw_pits != right.residual_state.raw_pits
        for left, right in zip(
            forward.model_states, reverse.model_states, strict=True
        )
    )
    for left, right in zip(forward.model_states, reverse.model_states, strict=True):
        np.testing.assert_allclose(
            [item.probability for item in left.posterior.members],
            [item.probability for item in right.posterior.members],
            atol=2e-14,
        )


def test_issued_family_is_isolated_from_unpassed_future_response() -> None:
    actions, targets = _history()
    engines = _engines()
    issued = reconstruct_likelihood_power_residual_family(
        engines, actions[:8], targets[:8]
    )
    stable_hash = issued.stable_hash
    raw_pits = tuple(state.residual_state.raw_pits for state in issued.model_states)
    changed = targets.copy()
    changed[8:] = -1e6
    reconstruct_likelihood_power_residual_family(engines, actions, changed)
    assert issued.stable_hash == stable_hash
    assert tuple(
        state.residual_state.raw_pits for state in issued.model_states
    ) == raw_pits


def test_production_binding_requires_exact_engine_and_posterior_objects() -> None:
    actions, targets = _history()
    family = reconstruct_likelihood_power_residual_family(
        _engines(), actions, targets
    )
    models = tuple(
        PosteriorModel(
            state.likelihood_power, state.engine, state.posterior
        )
        for state in family.model_states
    )
    assert _bound_semiparametric_residual_laws(models, family) == family.residual_laws
    copied = tuple(
        PosteriorModel(
            state.likelihood_power,
            state.engine,
            state.engine.fit_batch(actions, targets),
        )
        for state in family.model_states
    )
    with pytest.raises(ValueError, match="exact objects"):
        _bound_semiparametric_residual_laws(copied, family)


def test_family_erases_raw_history_after_state_reconstruction() -> None:
    actions, targets = _history()
    family = reconstruct_likelihood_power_residual_family(
        _engines(), actions, targets
    )
    assert not hasattr(family, "history_actions")
    assert not hasattr(family, "history_targets")
    source = inspect.getsource(implementation)
    assert "candidate_targets" not in source
    assert "np.random" not in source
    assert "result_status" not in source


def test_invalid_engine_family_fails_closed() -> None:
    actions, targets = _history()
    engines = _engines()
    with pytest.raises(ValueError, match="unique powers"):
        reconstruct_likelihood_power_residual_family(
            (engines[0], engines[0]), actions, targets
        )
    other_bank = generic_real_bank(1)
    other = SequentialReferencePosterior(other_bank, 0.75)
    with pytest.raises(ValueError, match="one bank"):
        reconstruct_likelihood_power_residual_family(
            (engines[0], other), actions, targets
        )


def test_one_step_posterior_update_is_exactly_batch_equivalent() -> None:
    actions, targets = _history()
    for engine in _engines():
        posterior = engine.prior_posterior()
        for index, (action, target) in enumerate(
            zip(actions, targets, strict=True), start=1
        ):
            posterior = engine.update_one(posterior, action, float(target))
            batch = engine.fit_batch(actions[:index], targets[:index])
            np.testing.assert_allclose(
                [member.probability for member in posterior.members],
                [member.probability for member in batch.members],
                atol=3e-14,
            )
            for left, right in zip(
                posterior.members, batch.members, strict=True
            ):
                np.testing.assert_allclose(
                    left.state.precision, right.state.precision, atol=3e-14
                )
                np.testing.assert_allclose(
                    left.state.information, right.state.information, atol=3e-14
                )


def test_conditioning_prefix_fits_base_but_never_enters_residual_history() -> None:
    actions, targets = _history()
    family = reconstruct_conditioned_likelihood_power_residual_family(
        _engines(), actions[:4], targets[:4], actions[4:8], targets[4:8]
    )
    assert family.conditioning_count == 4
    assert family.observation_count == 4
    assert all(
        len(state.residual_state.raw_pits) == 4 for state in family.model_states
    )
    for state in family.model_states:
        batch = state.engine.fit_batch(actions[:8], targets[:8])
        np.testing.assert_allclose(
            [member.probability for member in state.posterior.members],
            [member.probability for member in batch.members],
            atol=3e-14,
        )


def test_incremental_family_advance_matches_full_conditioned_reconstruction() -> None:
    actions, targets = _history()
    engines = _engines()
    initial = reconstruct_conditioned_likelihood_power_residual_family(
        engines, actions[:4], targets[:4], actions[4:8], targets[4:8]
    )
    advanced = advance_likelihood_power_residual_family(
        initial,
        actions[:4],
        targets[:4],
        actions[4:9],
        targets[4:9],
    )
    rebuilt = reconstruct_conditioned_likelihood_power_residual_family(
        engines, actions[:4], targets[:4], actions[4:9], targets[4:9]
    )
    assert advanced.stable_hash == rebuilt.stable_hash
    assert advanced.history_commitment == rebuilt.history_commitment
    for left, right in zip(
        advanced.model_states, rebuilt.model_states, strict=True
    ):
        assert left.residual_state.raw_pits == right.residual_state.raw_pits
        np.testing.assert_allclose(
            [member.probability for member in left.posterior.members],
            [member.probability for member in right.posterior.members],
            atol=3e-14,
        )


def test_incremental_advance_rejects_changed_prefix() -> None:
    actions, targets = _history()
    family = reconstruct_conditioned_likelihood_power_residual_family(
        _engines(), actions[:4], targets[:4], actions[4:8], targets[4:8]
    )
    changed = targets.copy()
    changed[5] += 0.1
    with pytest.raises(ValueError, match="prefix commitment changed"):
        advance_likelihood_power_residual_family(
            family,
            actions[:4],
            targets[:4],
            actions[4:9],
            changed[4:9],
        )


def test_frozen_p3h4_config_and_runner_evaluation_pass() -> None:
    config = runner._load_config(CONFIG)
    result = runner._evaluate(config)
    assert result["schema"] == runner.RESULT_SCHEMA
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert not result["p3h2_eta1_calibration_transferred_to_family"]
    assert not result["family_wide_calibration_established"]
    assert not result["operational_execution_authorized"]


def test_p3h4_config_tampering_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["state_mapping"]["nominal_state_copying"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="contract was modified"):
        runner._load_config(path)
