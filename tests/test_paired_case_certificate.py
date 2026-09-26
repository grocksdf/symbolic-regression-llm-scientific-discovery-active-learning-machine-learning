"""Artifact-only paired case certificate fixtures."""

import json
from pathlib import Path

from hypothesis_mvp.discovery.paired_case_certificate import (
    build_paired_case_certificate,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_paired_case_certificate_distinguishes_positive_and_boundary_cases(
        tmp_path):
    base = {
        "candidate_response_accessed": False, "heldout_opened": False}
    gas_v = {"variants": {
        "full": {"passed": True, "class_entropy_nats": .1,
                 "operational_class_count": 3},
        "no_llm": {"passed": False, "class_entropy_nats": 1e-20,
                   "operational_class_count": 1}}}
    gas_a = {"variants": {"full": {"sources": {
        "llm": {"admitted": True, "weight": .4,
                "fold_log_score_gains_vs_core": [2., 1.],
                "fold_numerical_tolerances": [1e-9, 1e-9]}}}}}
    yacht_v = {"variants": {
        "full": {"passed": False, "class_entropy_nats": 1e-20,
                 "operational_class_count": 1},
        "no_llm": {"passed": True, "class_entropy_nats": 1e-6,
                   "operational_class_count": 2}}}
    yacht_a = {"variants": {"full": {"sources": {
        "llm": {"admitted": False, "weight": 0.,
                "fold_log_score_gains_vs_core": [-1., -2.],
                "fold_numerical_tolerances": [1e-9, 1e-9]}}}}}
    for name, value in (
            ("gv.json", gas_v), ("ga.json", gas_a),
            ("yv.json", yacht_v), ("ya.json", yacht_a)):
        _write(tmp_path / name, value)
    result = build_paired_case_certificate(
        tmp_path / "gv.json", tmp_path / "ga.json",
        tmp_path / "yv.json", tmp_path / "ya.json")
    assert result["passed"] is True
    assert result["acquisition_efficacy_demonstrated"] is False
    assert result["heldout_superiority_demonstrated"] is False
