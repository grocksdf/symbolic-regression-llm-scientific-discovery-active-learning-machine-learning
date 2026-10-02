"""Algebraic correctness examples, not benchmark rescoring."""

from hypothesis_mvp.discovery.formula_recovery import assess_formula_recovery


def test_unbound_parameters_never_count_as_exact_recovery():
    result = assess_formula_recovery(
        "0.938*sin(x0)-0.867*sin(x1)",
        "F0*sin(t)-beta*sin(v)", ("t", "v"))
    assert result["structural_topology"] is True
    assert result["literal_exact"] is None
    assert result["parameter_instantiated_exact"] is None


def test_exact_instantiation_requires_registered_values():
    arguments = ("0.938*sin(x0)-0.867*sin(x1)",
                 "F0*sin(t)-beta*sin(v)", ("t", "v"))
    match = assess_formula_recovery(
        *arguments, parameter_bindings={"F0": .938, "beta": .867})
    mismatch = assess_formula_recovery(
        *arguments, parameter_bindings={"F0": .9, "beta": .867})
    assert match["literal_exact"] is None
    assert match["parameter_instantiated_exact"] is True
    assert mismatch["parameter_instantiated_exact"] is False
    assert mismatch["structural_topology"] is True


def test_internal_frequency_is_structural_parameter_slot():
    arguments = (".8*sin(1.7*x0)", "F0*sin(omega*t)", ("t",))
    structural = assess_formula_recovery(*arguments)
    instantiated = assess_formula_recovery(
        *arguments, parameter_bindings={"F0": .8, "omega": 1.7})
    assert structural["structural_topology"] is True
    assert structural["parameter_instantiated_exact"] is None
    assert instantiated["parameter_instantiated_exact"] is True


def test_registered_state_function_notation_maps_to_input_coordinate():
    result = assess_formula_recovery(
        "0.3*x1*sin(0.7*x0)",
        "k*P(t)*sin(0.7*t)",
        ("t", "P"))
    assert result["applicability"] == "unbound-parameters"
    assert result["literal_exact"] is None
    assert result["structural_topology"] is True
    assert result["unbound_truth_parameters"] == ["k"]
