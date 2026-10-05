"""Inference correctness diagnostic fixtures; no benchmark or efficacy data."""
from dataclasses import replace
import base64
import json
import numpy as np
import pytest

from hypothesis_mvp.discovery.candidate_validity import (
    CandidateValidityError, domain_obligations, numerical_definedness_preflight,
    proposal_domain_context,
)
from hypothesis_mvp.discovery.contracts import DiscoveryConfig
from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.evaluation_runtime import EvaluationRuntime
from hypothesis_mvp.discovery.pcpi_adapter import (
    EXPANDED_FORMULA_POLICY, freeze_expanded_formula_model, freeze_discovery_target,
)
from hypothesis_mvp.discovery.proposal_calibration import (
    assert_generation_context_safe, development_failure_feedback,
)
from hypothesis_mvp.discovery.proposal_runtime import ProposalCandidate, ProposalContext, ProposalRuntime
from hypothesis_mvp.discovery.regional_candidate_admission import _fixed_predictive_logpdf
from hypothesis_mvp.discovery.mechanism_funnel import candidate_mechanism_funnel
from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import (
    compile_fixed_formula_support, decode_fixed_formula_term, evaluate_fixed_formula_term,
)


@pytest.mark.parametrize('expression,points,reason', [
    ('log(x0)', [[0.]], 'domain_undefined'),
    ('log(x0)', [[-1.]], 'domain_undefined'),
    ('sqrt(x0)', [[-1.]], 'domain_undefined'),
    ('x0**.5', [[-1.]], 'domain_undefined'),
    ('x0**-.5', [[0.]], 'domain_undefined'),
    ('x0**-2', [[0.]], 'domain_undefined'),
    ('1/x0', [[0.]], 'domain_undefined'),
    ('x0/x0', [[0.]], 'domain_undefined'),
    ('exp(x0)', [[1000.]], 'numerical_nonfinite'),
    ('exp(x0)', [[-1000.]], 'numerical_nonfinite'),
    ('1e308*x0', [[2.]], 'numerical_nonfinite'),
    ('x0 + (1e-200*1e-200)', [[1.]], 'numerical_nonfinite'),
    ('x0 + 1e-400', [[1.]], 'numerical_nonfinite'),
    ('1e400*x0', [[1.]], 'numerical_nonfinite'),
])
def test_raw_expression_rejects_domain_and_intermediate_errors(expression, points, reason):
    with pytest.raises(CandidateValidityError) as caught:
        numerical_definedness_preflight(expression, np.array(points))
    assert caught.value.reason == reason


@pytest.mark.parametrize('expression', [
    'log(x0)', 'sqrt(x0)', 'x0**.25', 'x0**-2', '1/x0',
    'x0/(1+x1**2)', 'sin(1.7*x0)', 'exp(-x0)',
    'x0/(1+Abs(x1))', 'tanh(x0*x1)',
])
def test_valid_operator_families_remain_available(expression):
    points = np.array([[.25, -.5], [.5, 0.], [1., 2.]])
    audit = numerical_definedness_preflight(expression, points)
    assert audit['passed'] and not audit['response_accessed'] and not audit['semantics_repaired']
    support = compile_fixed_formula_support(expression, 2)
    assert all(np.all(np.isfinite(evaluate_fixed_formula_term(token, points))) for token in support)


@pytest.mark.parametrize('expression', [
    'x0/(1+x1**2)', 'x0/(1+Abs(x1))', 'sqrt(x0)', 'log(1+x0)',
    'sin(1.7*x0)', 'x0/(2*x1)', 'exp(-.7*x0)',
])
def test_full_amplitude_dag_roundtrip_preserves_inner_structure(expression):
    runtime = EquationRuntime(2, refit_policy='pcpi-expanded-fixed-inner-v1')
    before = compile_fixed_formula_support(expression, 2)
    after = compile_fixed_formula_support(runtime.dag('2.3*(' + expression + ')').expression, 2)
    assert before == after
    if '1.7' in expression:
        assert before != compile_fixed_formula_support(expression.replace('1.7', '1.8'), 2)


def test_legacy_tokens_replay_without_rewriting_frozen_bank():
    expression = 'x0 / (1 + x1 ** 2)'
    token = 'formula_ast_v1:' + base64.urlsafe_b64encode(expression.encode()).decode().rstrip('=')
    assert decode_fixed_formula_term(token, 2) == expression
    assert np.array_equal(evaluate_fixed_formula_term(token, np.array([[2., 1.]])), [1.])


def _evaluation():
    runtime = EquationRuntime(2, refit_policy='pcpi-expanded-fixed-inner-v1')
    config = DiscoveryConfig.from_mapping({'refit_policy': runtime.refit_policy,
                                           'evaluation_budget': 50, 'optimizer_maxiter': 20})
    return runtime, EvaluationRuntime(runtime, config)


def test_preflight_precedes_fit_and_checks_registered_action_points(monkeypatch):
    runtime, evaluation = _evaluation()
    points = np.array([[1., 1.], [2., 2.]])
    evaluation.registered_covariates = np.array([[0., 3.]])
    monkeypatch.setattr(runtime, 'refit_global_constants', lambda *a, **k: pytest.fail('fit opened'))
    assert evaluation.build_state('log(x0)', points, [1., 2.], points, [1., 2.],
        source='llm_evidence_synthesis', origin='llm', island='fixture', round_id=0) is None
    assert evaluation.rejections[-1]['reason'] == 'domain_undefined'
    assert evaluation.validity_attempts[-1]['compiled']
    assert not evaluation.validity_attempts[-1]['fit_survived']


def test_legitimate_rational_new_skeleton_survives_real_fit_and_retention():
    runtime, evaluation = _evaluation()
    points = np.column_stack([np.linspace(.3, 2., 24), np.linspace(-1., 1., 24)])
    target = 2.3 * points[:, 0] / (1 + points[:, 1]**2)
    parent = evaluation.build_state('x0', points, target, points, target,
        source='engine:fixture', origin='engine', island='fixture', round_id=0)
    proposal = ProposalCandidate(candidate_id='c', parent_hash=parent.dag.canonical_hash,
        island='fixture', action='PROPOSE_NEW_SKELETON', equation='x0/(1+x1**2)', rationale='fixture',
        lineage_id='valid-rational', prompt_hash='p', response_hash='r')
    candidate = evaluation.build_state(proposal.equation, points, target, points, target,
        source='llm_proposal', origin='llm', island='fixture', round_id=1, parent=parent, proposal=proposal)
    assert candidate is not None
    assert candidate.lineage[-1].retention['pass']
    assert np.allclose(runtime.predict(candidate.dag.expression, points), target, atol=1e-6)
    assert evaluation.validity_attempts[-1]['evaluated']


@pytest.mark.parametrize('expression', ['log(x0)', 'sqrt(x0)', 'x0**.25'])
def test_valid_positive_domain_survives_metric_probes(expression):
    runtime, evaluation = _evaluation()
    points = np.column_stack([np.linspace(.01, .5, 24), np.linspace(1., 2., 24)])
    target = runtime.predict(expression, points)
    candidate = evaluation.build_state(expression, points, target, points, target,
        source='llm_evidence_synthesis', origin='llm', island='fixture', round_id=0)
    assert candidate is not None, evaluation.rejections
    assert candidate.metrics.finite


@pytest.mark.parametrize('field', ['admission_score', 'admission_report', 'margin',
    'log_bayes_factors', 'region_log_predictive_ratios', 'region_selector_mass', 'log_evidence_threshold'])
def test_admission_fields_never_reach_generation_transport(field):
    runtime = ProposalRuntime(EquationRuntime(2, refit_policy='pcpi-expanded-fixed-inner-v1'), 2, None, 1)
    context = ProposalContext(1, 'fixture', 'p', 2, 1)
    with pytest.raises(ValueError, match='forbidden'):
        runtime._proposal_payload('fixture', 'fixture', context, {'nested': [{field: 1.}]}, (), ())


def test_feedback_whitelist_excludes_free_form_errors_scores_and_reused_rows():
    feedback = development_failure_feedback([
        {'origin': 'llm', 'rejection_reason': 'domain_undefined', 'error': 'secret'},
        {'origin': 'llm', 'rejection_reason': 'numerical_nonfinite', 'reused': True},
        {'origin': 'llm', 'rejection_reason': 'admission_failed', 'margin': -20},
    ])
    assert feedback['failure_counts'] == {'domain_undefined': 1}
    assert 'secret' not in json.dumps(feedback) and '-20' not in json.dumps(feedback)
    assert_generation_context_safe(feedback)
    domain = proposal_domain_context(np.array([[-1., 2.], [0., 3.]]))
    assert domain['variables'][0]['includes_zero']
    assert domain_obligations('log(x0) + x1**-.5 + x0/x1', 2)


def test_funnel_has_explicit_denominators_and_excludes_carried_candidates():
    rows = [{'origin': 'llm', 'source': 'llm_evidence_synthesis', 'compiled': True,
        'domain_valid': True, 'fit_survived': True, 'evaluated': True, 'structural_support': ['a']},
        {'origin': 'llm', 'source': 'llm_evidence_synthesis', 'compiled': True,
         'domain_valid': False, 'rejection_reason': 'domain_undefined'},
        {'origin': 'llm', 'source': 'llm_evidence_synthesis', 'reused': True}]
    report = {'candidate_validity_attempts': rows,
        'llm_rounds': [{'islands': [{'candidate_audit': [{'validated': True, 'accepted': False}]}]}]}
    admission = {'attempts': 2, 'candidates': [{'admitted': True, 'region_log_predictive_ratios': [8.]},
                                             {'admitted': False, 'reason': 'duplicate-engine-support'}]}
    result = candidate_mechanism_funnel(report, {'directive_count': 3}, admission)
    assert result['domain_valid_given_typed'] == {'numerator': 1, 'denominator': 2, 'rate': .5}
    assert result['fit_survives_given_compiled']['rate'] == .5
    assert result['independent_admitted_given_admission_evaluated']['denominator'] == 1
    assert candidate_mechanism_funnel({}, {})['domain_valid_given_typed']['rate'] is None


def test_admission_predictive_score_uses_native_log_weights_under_underflow():
    points = np.linspace(1., 2., 800)[:, None]
    fit = RoleDataset(DataRole.DEVELOPMENT, points, 2 * points[:, 0])
    observed = RoleDataset(DataRole.VALIDATION, np.array([[1000.]]), np.array([1e6]))
    rows = [{'expression': 'x0', 'source': 'engine:a'}, {'expression': 'x0**2', 'source': 'engine:b'}]
    score, _ = _fixed_predictive_logpdf(rows, fit, observed, points[::200], NormalInverseGammaPrior(), 2,
                                      coefficient_policy=EXPANDED_FORMULA_POLICY)
    model = freeze_expanded_formula_model(rows, n_features=1, prior=NormalInverseGammaPrior(),
        exploration_identity='f'*64, coefficient_policy=EXPANDED_FORMULA_POLICY, minimum_supports=1)
    target = freeze_discovery_target(model, fit, points[::200], measurement_budget=2,
                                    expected_model_identity=model.stable_hash)
    assert min(target.initial_posterior.log_probabilities) < np.log(np.nextafter(0., 1.))
    direct = model.engine(model.stable_hash).predictive_logpdf(target.initial_posterior, observed.X, observed.y)
    assert np.array_equal(score, direct)
    from scipy.special import logsumexp
    from scipy.stats import t as student_t
    from hypothesis_mvp.pcpi.acquisition import predictive_components_for_partition
    components = predictive_components_for_partition(model.engine(model.stable_hash),
        target.initial_posterior, target.partition, observed.X)
    clamped = logsumexp(np.log(components.structure_probabilities[:, None]) + student_t.logpdf(
        observed.y[None, :], df=components.degrees_freedom[:, None],
        loc=components.locations, scale=components.scales), axis=0)
    assert abs(clamped[0] - direct[0]) > 100


def test_funnel_keeps_distinct_valid_mechanism_families():
    expressions = ['x0*x1', 'x0/(1+x1**2)', 'log(x0)', 'sqrt(x0)']
    rows = [{'origin': 'llm', 'source': 'llm_evidence_synthesis', 'expression': value,
             'compiled': True, 'domain_valid': True, 'structural_support': [value]}
            for value in expressions]
    result = candidate_mechanism_funnel({'candidate_validity_attempts': rows}, {})
    assert result['distinct_valid_support_count'] == len(expressions)
    assert len(result['valid_operator_family_counts']) == len(expressions)


def test_formula_false_has_explicit_unresolved_proof_status():
    from hypothesis_mvp.discovery.formula_recovery import assess_formula_recovery
    result = assess_formula_recovery('sin(x0)**2+cos(x0)**2+x0', '1+x0', ('x0',))
    assert result['literal_exact'] is False
    assert result['literal_exact_status'] == 'unresolved'
    assert result['false_is_proof_of_inequivalence'] is False


def test_failed_quad_vec_convergence_fails_closed(monkeypatch):
    from types import SimpleNamespace
    import hypothesis_mvp.pcpi.acquisition as acquisition
    partition = acquisition.ClassPartition(('a', 'b'), ((0,), (1,)), (.5, .5), (0, 1))
    components = acquisition.PredictiveComponents(np.array([.5, .5]), np.array([5., 5.]),
        np.array([[0.], [1.]]), np.ones((2, 1)), partition)
    def failed(*args, **kwargs):
        assert kwargs['full_output'] is True
        return np.array([.6]), .1, SimpleNamespace(success=False)
    monkeypatch.setattr(acquisition, 'quad_vec', failed)
    with pytest.raises(FloatingPointError, match='converge'):
        acquisition.exact_class_decision_risk_reduction_shared_actions(components)


def test_numerical_error_estimate_is_not_reported_as_rigorous_certificate():
    from hypothesis_mvp.discovery.realized_drr import _query_record
    from hypothesis_mvp.pcpi.acquisition import ExactEIGResult
    exact = ExactEIGResult(np.array([.3]), np.array([.01]))
    row = _query_record(0, 0, .29, np.array([.31]), 0, .001,
                        'certified-decision-risk', True, exact, .5, .4)
    assert row['numerical_targeted_selection']
    assert not row['certified_targeted_selection']
    assert not row['rigorous_enclosure']
    assert row['selection_mode'] == 'numerical-decision-risk'


def test_generation_context_projection_excludes_audit_and_admission_artifacts():
    from hypothesis_mvp.discovery.proposal_calibration import generation_orchestration_context
    context = {'research_plan': {'purpose': 'fixture'},
               'evidence_synthesis': {'candidate_admission': 'abstain'},
               'admission_report': {'margin': 999}, 'posterior_gap_audit': {'responses': [42]}}
    assert generation_orchestration_context(context) == {'research_plan': {'purpose': 'fixture'}}


def test_failed_typed_attempt_keeps_its_registered_lineage():
    runtime, evaluation = _evaluation()
    points = np.array([[0., 1.], [1., 2.]])
    assert evaluation.build_state('log(x0)', points, [1., 2.], points, [1., 2.],
        source='llm_evidence_synthesis', origin='llm', island='typed_synthesis', round_id=0,
        audit_lineage_id='typed-lineage') is None
    assert evaluation.validity_attempts[-1]['lineage_id'] == 'typed-lineage'


def test_carried_seed_is_excluded_from_new_validity_population():
    runtime, evaluation = _evaluation()
    points = np.column_stack([np.linspace(.2, 1., 12), np.linspace(1., 2., 12)])
    rows = [{'expression': 'x0', 'source': 'prior-llm', 'origin': 'llm',
             'lineage_id': 'previous-cycle', 'reused_candidate': True}]
    evaluation.seed_survivor(rows, points, points[:, 0], points, points[:, 0])
    assert evaluation.validity_attempts[-1]['reused']
    assert candidate_mechanism_funnel({'candidate_validity_attempts': evaluation.validity_attempts}, {})[
        'fit_survives_given_compiled']['denominator'] == 0
