"""Response-free algebraic tests for the CERT.19 confidence repair."""

from __future__ import annotations

from fractions import Fraction
import math

import pytest

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA,
    P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM,
    P3F4_CERT19_ENVELOPE_KERNEL_INTEGRATION_AUTHORIZED,
    P3F4_CERT19_RUN_AUTHORIZED,
    ResidentDirectConfidencePlan,
    direct_confidence_failure_allocation,
    direct_confidence_particle_lower_bound,
    envelope_anchor_minorization_lower,
    marion_fixed_path_particle_lower_bound,
    minorization_mixing_steps,
)


def _plan(**overrides: object) -> ResidentDirectConfidencePlan:
    values: dict[str, object] = {
        "schema": P3F4_CERT19_DIRECT_CONFIDENCE_SCHEMA,
        "theorem": P3F4_CERT19_DIRECT_CONFIDENCE_THEOREM,
        "contract_hash": "contract",
        "feynman_kac_plan_hash": "feynman-kac",
        "operational_estimand_hash": "estimand",
        "class_projector_hash": "projector",
        "path_step_bound": 320,
        "relative_ess_floor": Fraction(4, 5),
        "map_regret_budget": Fraction(1, 50),
        "failure_probability": Fraction(1, 20),
        "maximum_rejuvenation_steps_per_bridge": 200,
    }
    values.update(overrides)
    return ResidentDirectConfidencePlan(**values)  # type: ignore[arg-type]


def test_direct_confidence_allocation_closes_the_failure_budget() -> None:
    plan = _plan()
    assert plan.coupling_step_failure == Fraction(1, 12_800)
    assert plan.concentration_failure == Fraction(1, 512_000)
    assert plan.derived_failure_upper == plan.failure_probability
    for step in range(1, plan.path_step_bound + 1):
        success_lower = (
            1
            - step * plan.coupling_step_failure
            - plan.concentration_failure / plan.coupling_step_failure
        )
        assert success_lower >= Fraction(3, 4)


def test_direct_confidence_particle_count_replaces_median_amplification() -> None:
    plan = _plan()
    legacy_per_island = marion_fixed_path_particle_lower_bound(
        plan.path_step_bound,
        float(plan.relative_ess_floor),
        float(plan.functional_error_tolerance),
    )
    assert legacy_per_island == 53_102
    assert plan.particle_count == 69_197
    assert plan.confirmation_island_count == 1
    assert plan.particle_count < 9 * legacy_per_island
    assert plan.maximum_confirmation_target_evaluations == 4_428_608_000


def test_particle_formula_is_dimension_free_and_monotone_in_alpha() -> None:
    common = {
        "path_step_bound": 320,
        "relative_ess_floor": Fraction(4, 5),
        "bounded_functional_error": Fraction(1, 100),
    }
    loose = direct_confidence_particle_lower_bound(
        **common,
        failure_probability=Fraction(1, 20),
    )
    strict = direct_confidence_particle_lower_bound(
        **common,
        failure_probability=Fraction(1, 100),
    )
    assert loose == 69_197
    assert strict > loose
    assert _plan(class_projector_hash="six-to-the-seven").particle_count == loose
    assert _plan(class_projector_hash="six-to-the-seven-hundred").particle_count == loose


def test_envelope_anchor_strictly_improves_the_prior_minorization() -> None:
    core_evidence = 0.1467437810166268
    likelihood_envelope = 5639.272478769489
    tail_mass = Fraction(8, 125)
    anchor = envelope_anchor_minorization_lower(
        math.log(core_evidence),
        math.log(likelihood_envelope),
        tail_mass,
    )
    prior = core_evidence / likelihood_envelope
    target = _plan().per_bridge_mixing_tv_target
    assert anchor > 15.0 * prior
    assert minorization_mixing_steps(anchor, target) < minorization_mixing_steps(
        prior,
        target,
    )


def test_direct_confidence_guards_fail_closed() -> None:
    assert not P3F4_CERT19_RUN_AUTHORIZED
    assert not P3F4_CERT19_ENVELOPE_KERNEL_INTEGRATION_AUTHORIZED
    with pytest.raises(ValueError, match="alpha <= 1/4"):
        direct_confidence_failure_allocation(10, Fraction(1, 3))
    with pytest.raises(ValueError, match="one selection and one confirmation"):
        _plan(confirmation_island_count=3)
    with pytest.raises(ValueError, match="claim boundary"):
        _plan(median_amplification_used=True)
