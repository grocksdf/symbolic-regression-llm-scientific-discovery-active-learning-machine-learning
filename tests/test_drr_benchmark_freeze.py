"""Metadata-only DRR benchmark selection correctness fixtures."""

import json

import pytest

from hypothesis_mvp.discovery import drr_benchmark_freeze as freeze


def test_drr_selection_is_deterministic_and_never_reads_samples(
        tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "lsr_bench_data.hdf5"
    with h5py.File(hdf5, "w") as target:
        for family, group_path in freeze.FAMILY_PATHS.items():
            group = target.create_group(group_path)
            for index in range(12):
                group.create_group(f"{family}-{index}")
    protocol = {
        "selection_seed": "fixture",
        "family_quotas": {
            "lsr_transform": 2, "bio_pop_growth": 2,
            "chem_react": 2, "matsci": 2},
        "excluded_tasks": {
            "bio_pop_growth": ["bio_pop_growth-0"]}}
    tasks = freeze._task_names(hdf5)
    first = freeze._selected_tasks(tasks, protocol)
    second = freeze._selected_tasks(tasks, protocol)
    assert first == second
    assert "bio_pop_growth-0" not in first["bio_pop_growth"]
    assert sum(map(len, first.values())) == 8


def test_primary_protocol_has_one_metric_and_all_failures_in_denominator():
    protocol = json.loads((
        freeze.Path(__file__).resolve().parents[1] / "configs" /
        "scientific_aistats_drr_protocol_v1.json").read_text(
            encoding="utf-8"))
    metric = protocol["primary_metric"]
    assert metric["name"] == "decision_readiness_rate"
    assert metric["all_registered_tasks_in_denominator"] is True
    assert metric["metric_switching_forbidden"] is True
    assert "failures" in metric["task_seed_indicator"]
