"""Inference correctness diagnostic fixtures; no experiment data is opened."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts/run_cbsi_zero_provider_experiment.py")
SPEC = importlib.util.spec_from_file_location("cbsi_scorer_fixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationTripwire:
    """Both evaluation inputs and labels stay inaccessible during search."""
    def __init__(self):
        self.opened = False
        self.x_reads = 0
        self.y_reads = 0

    @property
    def X(self):
        assert self.opened, "evaluation covariates accessed during search"
        self.x_reads += 1
        return np.asarray([[0.4], [0.8], [1.2]])

    @property
    def y(self):
        assert self.opened, "evaluation labels accessed during search"
        self.y_reads += 1
        return np.asarray([0.3, 0.7, 1.0])


def fixture_scorer(monkeypatch, evaluation=None):
    calls = []

    def fake_density(rows, selection, X, y, namespace):
        calls.append((len(rows), namespace["fold"]))
        base = -1.0 - np.square(np.asarray(y))
        return base if len(rows) == 2 else base + np.asarray([0.1, -0.2, 0.3])

    monkeypatch.setattr(MODULE, "density", fake_density)
    role = SimpleNamespace(X=np.asarray([[0.1], [0.5], [0.9]]),
                           y=np.asarray([0.2, 0.4, 0.8]))
    selection = SimpleNamespace(development=role)
    scorer = MODULE.CrossfitBankScorer(
        [{"expression": "x0", "source": "fixture"},
         {"expression": "x0**2", "source": "fixture"}],
        selection, role, role,
        EvaluationTripwire() if evaluation is None else evaluation,
        "inference-correctness", 7)
    return scorer, calls


def test_search_and_selection_never_open_evaluation(monkeypatch):
    scorer, calls = fixture_scorer(monkeypatch)
    first = scorer.crossfit_gain("sin(x0)")
    assert scorer.crossfit_gain("sin(x0)") == first
    assert calls == [(2, "left"), (2, "right"),
                     (3, "left"), (3, "right")]
    assert scorer.baseline_density_evaluations == 2
    assert scorer.candidate_density_evaluations == 2
    assert scorer.density_evaluations == 4
    assert scorer.score_calls == 2
    assert scorer.optimizer_fits == 2
    assert scorer.crossfit_cache_hits == scorer.cache_hits == 1
    with pytest.raises(ValueError, match="frozen selected"):
        scorer.evaluation_gain("sin(x0)")
    with pytest.raises(ValueError, match="scored search"):
        scorer.freeze_selection("cos(x0)")
    selection_id = scorer.freeze_selection("sin(x0)")
    assert len(selection_id) == 64
    assert scorer.freeze_selection("sin(x0)") == selection_id
    with pytest.raises(ValueError, match="already frozen"):
        scorer.freeze_selection("cos(x0)")
    with pytest.raises(ValueError, match="frozen selected"):
        scorer.evaluation_gain("cos(x0)")
    with pytest.raises(ValueError, match="search is closed"):
        scorer.crossfit_gain("sin(x0)")
    assert scorer.evaluation.x_reads == scorer.evaluation.y_reads == 0
    assert len(calls) == 4


def test_evaluation_opens_only_selected_expression_and_caches_once(monkeypatch):
    scorer, calls = fixture_scorer(monkeypatch)
    scorer.crossfit_gain("sin(x0)")
    scorer.freeze_selection("sin(x0)")
    scorer.evaluation.opened = True
    first = scorer.evaluation_gain("sin(x0)")
    original_weight = first[1][0]
    first[1][0] = 999.0
    first[2]["converged"] = False
    repeated = scorer.evaluation_gain("sin(x0)")
    assert repeated[1][0] == original_weight
    assert repeated[2]["converged"] is True
    assert calls[-2:] == [(2, "evaluation"), (3, "evaluation")]
    assert len(calls) == 6
    assert scorer.evaluation.x_reads == scorer.evaluation.y_reads == 2
    assert scorer.baseline_density_evaluations == 3
    assert scorer.candidate_density_evaluations == scorer.score_calls == 3
    assert scorer.density_evaluations == 6
    assert scorer.optimizer_fits == 3
    assert scorer.density_cache_hits == 2
    assert scorer.evaluation_cache_hits == 1


def test_baseline_fallback_closes_search_without_evaluation_access(monkeypatch):
    scorer, calls = fixture_scorer(monkeypatch)
    scorer.freeze_selection(None)
    with pytest.raises(ValueError, match="frozen selected"):
        scorer.evaluation_gain("x0")
    with pytest.raises(ValueError, match="already frozen"):
        scorer.freeze_selection("x0")
    assert scorer.evaluation.x_reads == scorer.evaluation.y_reads == 0
    assert len(calls) == 2


def test_exact_expression_caches_do_not_alias_new_candidates(monkeypatch):
    scorer, calls = fixture_scorer(monkeypatch)
    scorer.crossfit_gain("sin(x0)")
    scorer.crossfit_gain("(sin(x0))")
    assert scorer.candidate_density_evaluations == 4
    assert scorer.optimizer_fits == 4
    assert len(calls) == 6


def test_failed_density_attempt_is_counted_but_not_cached(monkeypatch):
    scorer, _ = fixture_scorer(monkeypatch)

    def fail(*args, **kwargs):
        raise FloatingPointError("fixture density failed")

    monkeypatch.setattr(MODULE, "density", fail)
    with pytest.raises(FloatingPointError, match="fixture density failed"):
        scorer.crossfit_gain("sin(x0)")
    assert scorer.candidate_density_evaluations == 1
    assert scorer.density_evaluations == 3
    assert scorer.score_calls == scorer.optimizer_fits == 0
    assert "sin(x0)" not in scorer._crossfit_cache
    assert ("sin(x0)", "left") not in scorer.cache


def test_cached_values_match_uncached_real_density_and_stacking():
    from hypothesis_mvp.data.roles import DataRole, RoleDataset

    def role(offset):
        x = np.asarray([[0.15], [0.45], [0.8], [1.1]]) + offset
        return RoleDataset(DataRole.DEVELOPMENT, x,
                           0.4 + x[:, 0] + 0.15 * np.square(x[:, 0]))

    development, left, right, evaluation = [role(v) for v in (0, .02, .04, .06)]
    selection = SimpleNamespace(
        development=development,
        acquisition_pool=SimpleNamespace(X=np.asarray([[.3], [.7], [1.0]])))
    baseline = [{"expression": "x0", "source": "fixture"},
                {"expression": "x0**2", "source": "fixture"}]
    scorer = MODULE.CrossfitBankScorer(
        baseline, selection, left, right, evaluation,
        "inference-correctness", 13)
    expression = "x0 + x0**2"
    candidate = {"expression": expression, "source": "cbsi:structural_edit",
                 "origin": "archive-edit", "lineage_id": "fixture"}
    uncached_base = {}
    uncached_candidate = {}
    for fold, values in (("left", left), ("right", right),
                         ("evaluation", evaluation)):
        uncached_base[fold] = MODULE.density(
            baseline, selection, values.X, values.y,
            {"fixture": "reference-base", "fold": fold})
        uncached_candidate[fold] = MODULE.density(
            [*baseline, candidate], selection, values.X, values.y,
            {"fixture": "reference-candidate", "fold": fold})
    # The default finite-bank prior gives one mass per unique support.
    # Repeating an existing support within its source family is a true no-op,
    # including when the expression has a different outer amplitude.
    duplicate_density = MODULE.density(
        [*baseline, {"expression": "2*x0", "source": "fixture"}],
        selection, left.X, left.y, {"fixture": "reference-duplicate"})
    np.testing.assert_array_equal(duplicate_density, uncached_base["left"])
    expected_search = tuple(MODULE.CrossfitBankScorer._mixture_gain(
        uncached_base[cal], uncached_candidate[cal],
        uncached_base[score], uncached_candidate[score], 13 + offset)[0]
        for cal, score, offset in (("left", "right", 1), ("right", "left", 2)))
    assert scorer.crossfit_gain(expression) == pytest.approx(expected_search)
    before = (scorer.density_evaluations, scorer.optimizer_fits)
    assert scorer.crossfit_gain(expression) == pytest.approx(expected_search)
    assert (scorer.density_evaluations, scorer.optimizer_fits) == before
    scorer.freeze_selection(expression)
    expected_eval = MODULE.CrossfitBankScorer._mixture_gain(
        np.concatenate([uncached_base["left"], uncached_base["right"]]),
        np.concatenate([uncached_candidate["left"], uncached_candidate["right"]]),
        uncached_base["evaluation"], uncached_candidate["evaluation"], 16)
    actual_eval = scorer.evaluation_gain(expression)
    assert actual_eval[0] == pytest.approx(expected_eval[0])
    np.testing.assert_allclose(actual_eval[1], expected_eval[1])
    before = (scorer.density_evaluations, scorer.optimizer_fits)
    assert scorer.evaluation_gain(expression) == actual_eval
    assert (scorer.density_evaluations, scorer.optimizer_fits) == before


def test_existing_bank_support_is_an_exact_noop(monkeypatch):
    scorer, calls = fixture_scorer(monkeypatch)
    assert scorer.crossfit_gain("2*x0") == pytest.approx((0.0, 0.0), abs=1e-12)
    scorer.freeze_selection("2*x0")
    scorer.evaluation.opened = True
    assert scorer.evaluation_gain("2*x0")[0] == pytest.approx(0.0, abs=1e-12)
    assert calls == [(2, "left"), (2, "right"), (2, "evaluation")]
    assert scorer.candidate_density_evaluations == scorer.score_calls == 0
