"""P3F.3 particle-interface tests; correctness-only and never efficacy evidence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hypothesis_mvp.pcpi.open_target import (
    CountablyOpenTypedGrammar,
    OpenTargetContract,
    OpenTargetParticleConfig,
    ScalableOpenTargetSMC,
    proposal_invariance_certificate,
    sample_open_prior_expression,
)
from hypothesis_mvp.pcpi.reference import (
    DiscrepancyKernelState,
    NormalInverseGammaPrior,
    StructurewiseDiscrepancyPrior,
)


def _contract(maximum_nodes: int = 3) -> OpenTargetContract:
    return OpenTargetContract(
        CountablyOpenTypedGrammar(1, 0.4),
        maximum_nodes,
        NormalInverseGammaPrior(0.0, 0.7, 3.0, 0.08),
        StructurewiseDiscrepancyPrior(0.3, 1.2),
        (
            DiscrepancyKernelState("short", 0.5, 0.6),
            DiscrepancyKernelState("long", 0.5, 1.3),
        ),
    )


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([-1.25, -0.8, -0.35, 0.0, 0.3, 0.75, 1.2])[:, None]
    y = 0.15 + 0.7 * x[:, 0] + 0.2 * np.square(x[:, 0])
    return x, y


def test_p3f3_prior_sampler_respects_registered_slice() -> None:
    grammar = CountablyOpenTypedGrammar(1, 0.4)
    rng = np.random.default_rng(20260817)
    expressions = [
        sample_open_prior_expression(grammar, rng, maximum_nodes=3)
        for _ in range(512)
    ]
    assert all(1 <= expression.node_count <= 3 for expression in expressions)
    assert all(expression.expression_type == "dimensionless-real" for expression in expressions)


def test_p3f3_config_rejects_unregistered_proposal() -> None:
    with pytest.raises(ValueError, match="prior-independence"):
        OpenTargetParticleConfig(proposal_kind="llm")


def test_p3f3_config_rejects_unregistered_resampling_schedule() -> None:
    with pytest.raises(ValueError, match="pre-bridge or post-bridge"):
        OpenTargetParticleConfig(resampling_schedule="adaptive")


def test_p3f3_mixture_config_has_fixed_registered_weight() -> None:
    config = OpenTargetParticleConfig(proposal_kind="prior-uniform-mixture")
    assert config.proposal_mixture_weight == pytest.approx(0.5)


def test_p3f3_registered_proposals_have_exact_invariance_certificate() -> None:
    actions, targets = _fixture()
    certificate = proposal_invariance_certificate(
        _contract(), actions, targets, maximum_nodes=3
    )
    assert certificate["component_count"] == 42
    assert certificate["prefix_count"] == len(targets) + 1
    assert certificate["proposal_mixture_weight"] == pytest.approx(0.5)
    assert "prior-uniform-mixture" in certificate["proposal_kinds"]
    assert certificate["maximum_error"] < 2e-14


def test_p3f3_finite_slice_cutoff_is_part_of_the_target() -> None:
    with pytest.raises(ValueError, match="registered reference slice"):
        ScalableOpenTargetSMC(
            _contract(maximum_nodes=3),
            OpenTargetParticleConfig(particle_count=16, maximum_nodes=4),
            seed=2026081702,
        )


def test_p3f3_particle_result_records_normalized_mass_and_genealogy() -> None:
    actions, targets = _fixture()
    result = ScalableOpenTargetSMC(
        _contract(),
        OpenTargetParticleConfig(
            particle_count=128,
            maximum_nodes=3,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=0,
        ),
        seed=2026081701,
    ).run(actions, targets)
    probabilities = np.asarray(
        [particle.posterior_probability for particle in result.particles]
    )
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=2e-12)
    assert len(result.diagnostics) >= len(targets)
    # diagnostics 按观测 step 平铺：一个 step 可对应多条 bridge 记录，
    # 记录按 step 升序排列，且每个 step 至少一条（首条从 bridge_step=1 开始）。
    assert result.diagnostics[0].step == 1
    assert result.diagnostics[-1].step == len(targets)
    steps = [item.step for item in result.diagnostics]
    assert steps == sorted(steps)
    for step in range(1, len(targets) + 1):
        records = [item for item in result.diagnostics if item.step == step]
        assert records
        assert records[0].bridge_step == 1
        assert all(item.step == step for item in records)
    for diagnostic in result.diagnostics:
        assert len(diagnostic.ancestor_indices) == len(result.particles)
        assert len(diagnostic.parent_particle_ids) == len(result.particles)
        assert len(diagnostic.child_particle_ids) == len(result.particles)
        assert diagnostic.distinct_root_ancestors >= 1
        assert diagnostic.effective_sample_size_after >= 1.0
    for step in range(1, len(targets) + 1):
        bridges = [item for item in result.diagnostics if item.step == step]
        assert bridges[-1].beta_current == pytest.approx(1.0)
        assert all(
            previous.beta_current < current.beta_current
            for previous, current in zip(bridges, bridges[1:])
        )
    assert math.isclose(
        sum(result.equivalence_class_posterior.values()),
        1.0,
        abs_tol=2e-12,
    )


def test_p3f3_move_audit_is_aligned_with_rejuvenation_diagnostics() -> None:
    actions, targets = _fixture()
    result = ScalableOpenTargetSMC(
        _contract(),
        OpenTargetParticleConfig(
            particle_count=128,
            maximum_nodes=3,
            ess_threshold_fraction=0.5,
            rejuvenation_steps=1,
            proposal_kind="complete-uniform",
        ),
        seed=2026081704,
    ).run(actions, targets)
    expected = sum(item.proposals for item in result.diagnostics)
    assert len(result.moves) == expected
    assert sum(move.accepted for move in result.moves) == sum(
        item.acceptances for item in result.diagnostics
    )
    assert all(move.proposal_kind == "complete-uniform" for move in result.moves)
    assert all(move.proposal_component == "complete-uniform" for move in result.moves)
    assert all(move.ast_structural_distance >= 0 for move in result.moves)
    assert all(move.semantic_polynomial_l1_distance >= 0.0 for move in result.moves)
    assert all(move.observation_step >= 1 for move in result.moves)
    assert all(move.bridge_step >= 1 for move in result.moves)
    assert all(
        move.move_type
        in {
            "self-transition",
            "within-equivalence-class",
            "cross-equivalence-class",
            "discrepancy-state-change",
            "cross-equivalence-and-state-change",
        }
        for move in result.moves
    )
