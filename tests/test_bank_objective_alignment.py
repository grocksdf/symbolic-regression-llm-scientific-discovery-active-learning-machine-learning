from scripts.audit_bank_objective_alignment import mixture_counterexample, audit_admission


def test_proper_laws_show_standalone_gate_can_miss_mixture_value():
    example = mixture_counterexample()
    assert abs(sum(example["core"]) - 1) < 1e-12
    assert abs(sum(example["candidate"]) - 1) < 1e-12
    assert example["standalone_mean_log_ratio"] < 0
    assert example["mixture_mean_log_ratio"] > 0


def test_nonzero_weight_does_not_imply_useful_candidate():
    candidate = {"admitted": False, "global_log_predictive_ratio": -100,
                 "log_evidence_threshold": 4, "minimum_region_log_predictive_ratio": -120}
    row = {"task": "fixture", "seed": 1, "candidate_count": 1,
           "admitted_candidate_count": 0, "admission_audit": {"candidates": [candidate]},
           "candidate_weights": [{"weight": .9, "admitted_by_protected": False}],
           "gains": {"rejected_only_stacking": 1e-15, "full_stacking_matched": 0,
                     "protected_bma": 0}}
    result = audit_admission({"rows": [row, {"coordinate_failure": True}]}, 1e-8)
    assert result["counts"]["rejected_weight_without_material_gain"] == 1
    assert result["counts"]["both_global_and_region_fail"] == 1
    assert result["counts"]["historical_failed_coordinates"] == 1
