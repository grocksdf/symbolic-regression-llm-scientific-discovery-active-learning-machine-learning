"""Inference correctness diagnostic fixtures; no benchmark rescoring."""
from scripts.formula_recovery_contract import assess_formula_recovery


def test_external_amplitudes_do_not_become_a_false_exact_claim():
    result = assess_formula_recovery(
        "0.938*sin(x0)-0.867*sin(x1)",
        "F0*sin(t)-beta*sin(v)", ("t", "v"))
    assert result["structural_topology"] is True
    assert result["literal_exact"] is None
    assert result["unbound_truth_parameters"] == ["F0", "beta"]
    assert result["recovery_success_claimed"] is False


def test_internal_frequency_parameter_changes_structure():
    result = assess_formula_recovery("0.938*sin(x0)",
                                     "F0*sin(omega0*t)", ("t",))
    assert result["structural_topology"] is False
    assert result["literal_exact"] is None
    fitted = assess_formula_recovery("0.938*sin(1.7*x0)",
                                     "F0*sin(omega0*t)", ("t",))
    assert fitted["structural_topology"] is True
    assert fitted["parameter_instantiated_exact"] is None


def test_ratio_fractional_power_and_function_topology():
    same = assess_formula_recovery("x0/(1+x0**2)",
                                   "t/(1+t**2)", ("t",))
    assert same["literal_exact"] is True
    assert same["structural_topology"] is True
    wrong = assess_formula_recovery("sqrt(x0)", "log(t)", ("t",))
    assert wrong["structural_topology"] is False
    offset = assess_formula_recovery("log(Abs(x0)+2)",
                                     "beta*log(Abs(t)+1)", ("t",))
    assert offset["structural_topology"] is True
    assert offset["literal_exact"] is None


def test_unparseable_metadata_is_not_scored_as_zero():
    result = assess_formula_recovery("sin(x0)", "0.157_t*sin(t)", ("t",))
    assert result["applicability"] == "not-evaluable"
    assert result["literal_exact"] is None
    assert result["structural_topology"] is None


def test_unknown_candidate_symbol_cannot_match_as_a_parameter():
    result = assess_formula_recovery("alpha*sin(x0)",
                                     "alpha*sin(t)", ("t",))
    assert result["applicability"] == "not-evaluable"
    assert result["structural_topology"] is None
