"""Synthetic algebra/isolation checks only; no efficacy evidence."""
from dataclasses import replace
import ast

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.bank_search_feedback import BankSearchFeedback
from hypothesis_mvp.discovery.counterfactual_bank_symbolic_invention import (
    StructuralEditingEngine, _replace_at_path,
)
from hypothesis_mvp.discovery.region_site_attribution import (
    FEATURE_NAMES, build_region_site_contexts, subtree_response,
)
from hypothesis_mvp.discovery.regional_site_operator_policy import (
    RegionalSiteOperatorPolicy, SiteOperatorAction,
)
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import _run


@pytest.mark.parametrize("expression", ["x0*x1 + sin(x0)", "exp(x0)/(1+x1**2)",
    "log(x0)+sqrt(x1)", "tanh(x0)-abs(x1)", "x0**0 + x1", "x0**(-2) + x1"])
def test_all_edit_sites_analytic_derivative_matches_intervention(expression):
    x = np.column_stack((np.linspace(.6, 1.4, 9), np.linspace(.7, 1.8, 9)))
    for path in StructuralEditingEngine(2).editable_paths(expression):
        parent, values, derivative = subtree_response(expression, path, x)
        plus, minus = [], []
        epsilon = 1e-6
        for row, value in zip(x, values):
            for delta, target in ((epsilon, plus), (-epsilon, minus)):
                node = _replace_at_path(ast.parse(expression, mode="eval").body, path,
                                        ast.Constant(value=float(value + delta)))
                target.append(float(np.asarray(_run(node, row[None, :])).reshape(-1)[0]))
        np.testing.assert_allclose(derivative, (np.asarray(plus)-minus)/(2*epsilon), rtol=1e-5, atol=1e-7)
        assert np.isfinite(parent).all()


def test_undefined_derivative_and_unsafe_syntax_fail_closed():
    x = np.asarray([[0., 1.], [1., 2.]])
    with pytest.raises(ValueError, match="undefined"):
        subtree_response("abs(x0)", (1,), x)
    with pytest.raises(ValueError):
        subtree_response("__import__('os').system('x')", (), x)
    with pytest.raises(ValueError):
        subtree_response("x0/(x1-x1)", (), x)
    # Parent-output intervention is differentiable even at an inner cusp.
    np.testing.assert_equal(subtree_response("abs(x0)", (), x)[2], [1,1])


def context_fixture():
    x = np.column_stack((np.linspace(.4, 1.4, 12), np.linspace(.5, 2., 12)**2))
    fit = RoleDataset(DataRole.DEVELOPMENT, x, x[:,0]+np.sin(x[:,1]))
    residual = fit.y-x[:,0]
    feedback = BankSearchFeedback("bank", tuple(f"fit:{i}" for i in range(len(x))), fit.fingerprint,
        tuple(-residual**2), tuple(residual), tuple(residual), "[]")
    kwargs = dict(regions=[{"id":"r", "axis":1, "side":"gt", "cut":float(np.median(x[:,1]))}],
        forbidden_row_ids=("admission:0",), bank_location=x[:,0], bank_scale=np.ones(len(x)),
        parent_mass=.5, parent_redundancy=.2)
    return fit, feedback, kwargs


def test_context_contains_real_local_behavior_not_only_ast():
    fit, feedback, kw = context_fixture()
    result = build_region_site_contexts("x0 + sin(x1)", fit, feedback, **kw)
    leaves = [r for r in result["sites"] if r["subtree_expression"] in {"x0", "x1"}]
    assert len(leaves) == 2
    assert leaves[0]["features"] != leaves[1]["features"]
    assert len(result["sites"][0]["features"]) == len(FEATURE_NAMES)
    assert all(np.isfinite(r["features"]).all() for r in result["sites"])
    changed = build_region_site_contexts("x0 + sin(x1)", fit, feedback, **{**kw,"bank_location":fit.y+2})
    assert result["identity"] != changed["identity"]
    assert not result["causal_attribution_authorized"]


def test_independent_rows_cannot_supply_attribution_context():
    fit, feedback, kw = context_fixture()
    with pytest.raises(ValueError, match="isolated"):
        build_region_site_contexts("x0+x1", fit, feedback,
            **{**kw, "forbidden_row_ids":(feedback.search_row_ids[0],)})
    with pytest.raises(ValueError, match="isolated"):
        build_region_site_contexts("x0+x1", RoleDataset(DataRole.VALIDATION, fit.X, fit.y), feedback, **kw)


def test_intercept_only_bank_parent_has_explicit_diagnostic_context():
    fit, feedback, kw = context_fixture()
    snapshot = build_region_site_contexts("(-4.5)", fit, feedback, **kw)
    assert snapshot["sites"] and not snapshot["unavailable"]
    assert snapshot["sites"][0]["variables"] == []
    parent, subtree, jacobian = subtree_response("(-4.5)", (), fit.X)
    np.testing.assert_equal(parent, -4.5)
    np.testing.assert_equal(jacobian, 1.)


def test_local_derivative_does_not_discard_valid_region_due_to_outside_cusp():
    fit = RoleDataset(DataRole.DEVELOPMENT, [[0.], [1.], [4.]], [1., 2., 3.])
    feedback = BankSearchFeedback("bank", ("fit:0","fit:1","fit:2"), fit.fingerprint,
                                  (-1.,-1.,-1.), (1.,1.,1.), (1.,1.,1.), "[]")
    snapshot = build_region_site_contexts("sqrt(x0)", fit, feedback,
        regions=[{"id":"high", "axis":0,"side":"gt","cut":1.},
                 {"id":"low", "axis":0,"side":"le","cut":1.}],
        forbidden_row_ids=("adm:0",), bank_location=[0.,1.,2.], bank_scale=[1.,1.,1.],
        parent_mass=1., parent_redundancy=0.)
    assert any(r["region_id"]=="high" and r["site_path"]==[1] for r in snapshot["sites"])
    assert any(r["region_id"]=="low" and r["site_path"]==[1] for r in snapshot["unavailable"])


def actions():
    fit, feedback, kw = context_fixture()
    snapshot = build_region_site_contexts("x0 + sin(x1)", fit, feedback, **kw)
    return [SiteOperatorAction(snapshot["parent"], "x0*x1", "r", tuple(r["site_path"]), op,
                              snapshot["identity"], tuple(r["features"]))
            for r in snapshot["sites"] for op in ("correction_term", "subtree_replacement")], fit


def materialize(pair, fit):
    editor = StructuralEditingEngine(2)
    requests, results = [], []
    for a in pair:
        request = editor.deterministic_request(a.operator, a.parent, a.donor, seed=9,
            task_identity="task:block", site_path=a.site_path,
            matched_control_of=requests[0].identity if requests else None)
        requests.append(request)
        results.append(editor.apply(request, registered_domains=[fit.X],
            known_structural_hashes=set(), known_canonical_hashes=set(), known_support_hashes=set()))
    assert all(r.valid and r.novel for r in results)
    return dict(primary_request=requests[0], control_request=requests[1],
        primary_result=results[0], control_result=results[1], primary_folds=(1.,3.), control_folds=(0.,0.),
        evidence_id="pair", scoring_identity="scorer")


@pytest.mark.parametrize("contrast", ["site", "operator"])
def test_joint_policy_uses_both_intervention_types_and_one_fold_mean(contrast):
    pool, fit = actions()
    p = RegionalSiteOperatorPolicy(operators=("correction_term", "subtree_replacement"), seed=9,
                                   scoring_identity="scorer", task_identity="train")
    pair = p.choose_pair(0, pool, contrast=contrast)
    a,b = pair
    assert (a.site_path != b.site_path) == (contrast == "site")
    assert (a.operator != b.operator) == (contrast == "operator")
    evidence = materialize(pair, fit)
    p.observe_pair(0, **evidence)
    audit = p.audit()
    assert audit["posterior"]["alpha"] == 1.5
    assert audit["observations"][0]["contrast_mean"] == 2.
    assert (p.feature(a)-p.feature(b)) @ np.asarray(audit["posterior"]["mean"]) > 0
    with pytest.raises(ValueError, match="duplicate"):
        p.observe_pair(0, **evidence)
    assert p.audit() == audit


def test_atomic_identity_failure_and_explicit_transfer_isolation():
    pool, fit = actions()
    p = RegionalSiteOperatorPolicy(operators=("correction_term", "subtree_replacement"), seed=9,
                                   scoring_identity="scorer", task_identity="train")
    pair = p.choose_pair(0, pool, contrast="site")
    evidence = materialize(pair, fit)
    before = p.audit()
    with pytest.raises(ValueError, match="identity"):
        p.observe_pair(0, **{**evidence, "control_request":replace(evidence["control_request"], seed=99)})
    assert p.audit() == before
    with pytest.raises(ValueError, match="finite"):
        p.observe_pair(0, **{**evidence, "primary_folds":(np.nan, 1.)})
    assert p.audit() == before
    p.observe_pair(0, **evidence)
    prior = p.freeze_transfer_prior()
    q = RegionalSiteOperatorPolicy(operators=p.operators, seed=9, scoring_identity="new", task_identity="new", frozen_prior=prior)
    assert q.audit()["posterior"] == p.audit()["posterior"]
    assert q.audit()["observation_count"] == 0
    with pytest.raises(ValueError, match="isolation"):
        RegionalSiteOperatorPolicy(operators=p.operators, seed=9, scoring_identity="new", task_identity="train", frozen_prior=prior)
    prior["posterior"]["alpha"] = 99
    assert q.audit()["posterior"]["alpha"] == 1.5
    with pytest.raises(ValueError, match="identity"):
        RegionalSiteOperatorPolicy(operators=p.operators, seed=9, scoring_identity="new", task_identity="fresh", frozen_prior=prior)


def test_prior_cannot_cross_intervention_or_utility_and_equivalent_pairs_do_not_learn():
    pool, fit = actions()
    p = RegionalSiteOperatorPolicy(operators=("subtree_replacement",), seed=3, scoring_identity="scorer",
                                   task_identity="train", intervention_identity="E")
    pair = p.choose_pair(0, [a for a in pool if a.operator=="subtree_replacement"], contrast="site")
    evidence = materialize(pair, fit)
    p.observe_pair(0, **evidence)
    prior = p.freeze_transfer_prior()
    for changes in ({"intervention_identity":"D"}, {"intervention_identity":"E", "utility_identity":"other"}):
        with pytest.raises(ValueError, match="schema"):
            RegionalSiteOperatorPolicy(operators=p.operators, seed=4, scoring_identity="fresh", task_identity="fresh",
                frozen_prior=prior, **changes)
    q = RegionalSiteOperatorPolicy(operators=p.operators, seed=3, scoring_identity="scorer", task_identity="train")
    q.choose_pair(0, [a for a in pool if a.operator=="subtree_replacement"], contrast="site")
    with pytest.raises(ValueError, match="distinct"):
        q.observe_pair(0, **{**evidence, "control_result":replace(evidence["control_result"], support_hash=evidence["primary_result"].support_hash)})
    assert q.audit()["observation_count"] == 0


def test_same_ast_class_actions_have_distinct_shared_features_and_determinism():
    pool, _ = actions()
    leaf_actions = [a for a in pool if a.features[FEATURE_NAMES.index("leaf")] and a.operator == "correction_term"]
    p = RegionalSiteOperatorPolicy(operators=("correction_term",), seed=2, scoring_identity="s", task_identity="t")
    assert np.any(p.feature(leaf_actions[0]) != p.feature(leaf_actions[1]))
    q = RegionalSiteOperatorPolicy(operators=("correction_term",), seed=2, scoring_identity="s", task_identity="t")
    assert p.choose_pair(0, leaf_actions, contrast="site") == q.choose_pair(0, leaf_actions[::-1], contrast="site")
