"""Correctness and isolation checks for Expanded Formula Discovery."""

import numpy as np

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from scripts import expanded_formula_discovery_harness as subject


def _role(role, offset):
    X = np.arange(offset, offset + 8, dtype=float)[:, None]
    return RoleDataset(role, X, 2.0 * X[:, 0] + 1.0)


def test_formula_recovery_definitions():
    assert subject.exact_formula_recovery(
        "2*x0 + 1", "1 + 2*t", ["t"])
    assert not subject.exact_formula_recovery(
        "2*x0", "1 + 2*t", ["t"])
    assert subject.exact_formula_recovery(
        "x1*exp(-x0)", "P(t)*exp(-t)", ["t", "P(t)"])
    assert subject.structural_formula_recovery(
        "3*x0**2 + 7*sin(x0)", "9*t**2 + 2*sin(t)", ["t"])
    assert not subject.structural_formula_recovery(
        "3*x0 + 7*sin(x0)", "9*t**2 + 2*sin(t)", ["t"])


def test_three_admission_arms_share_candidates_and_protect_core(monkeypatch):
    calls = []

    def fake_filter(rows, fit, role, **kwargs):
        calls.append((tuple(dict(row) for row in rows), role.fingerprint))
        retained = [dict(row) for row in rows
                    if row.get("origin") != "llm"
                    or role.y.mean() > 20]
        return retained, {"role": role.fingerprint}

    monkeypatch.setattr(
        subject, "filter_fold_safe_source_candidates", fake_filter)
    core = [
        {"expression": "1", "source": "deterministic_constant_anchor",
         "origin": "deterministic"},
        {"expression": "x0", "source": "engine:polynomial_lasso",
         "origin": "deterministic"},
    ]
    optional = [
        {"expression": "x0**2", "source": "llm", "origin": "llm"},
        {"expression": "2*x0", "source": "llm", "origin": "llm"},
    ]
    result = subject.project_admission_arms(
        core, optional, _role(DataRole.DEVELOPMENT, 0),
        _role(DataRole.VALIDATION, 10),
        _role(DataRole.VALIDATION, 20),
        _role(DataRole.VALIDATION, 30),
        np.asarray([[0.], [1.]]), n_features=1, identity="a" * 64)
    assert tuple(result["arms"]) == subject.ADMISSION_ARMS
    assert all(rows[:2] == core for rows in result["arms"].values())
    assert len(result["arms"]["accept_all"]) == 3
    assert result["generative_authority_equals_epistemic_authority"] is False
    assert all(row["source"].startswith(
        "protected_counterfactual_backbone:") for row in calls[0][0][:2])
