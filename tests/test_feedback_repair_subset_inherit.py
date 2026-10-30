"""Subset execution and inheritance must not re-call the provider for kept children.

These tests exercise scripts.run_feedback_repair.resolve_execution_plan and
_inherit_child without any scientific provider, runner subprocess, or real
artifacts. They guard the "do not hardcode the full 72; allow a rerun subset
that inherits prior successful children" contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_feedback_repair as runner


def _make_artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = json.loads("""{
        "scientific_discovery_runtime": {
            "scientist_agent_system_evaluation": {
                "iterative_feedback_trace": [
                    {"bank_rows_after": [{"expression": "x0", "source": "core"}]},
                    {"bank_rows_after": [{"expression": "x0", "source": "core"}]}
                ]
            },
            "scientist_agent_engine_reports": [{
                "run_records": [
                    {"engine": "pysr", "repeat": 0, "attempt": 0, "seed": 1,
                     "lineage_id": "p", "status": "succeeded"},
                    {"engine": "polynomial_lasso", "repeat": 0, "attempt": 0, "seed": 2,
                     "lineage_id": "l", "status": "succeeded"}
                ]
            }]
        }
    }""")
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def _make_prior_execution(root, labels, successful):
    prior = root / "v2exec"
    for label in labels:
        folder = prior / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        if label in successful:
            artifact = folder / "pcpi_artifacts" / "art.json"
            _make_artifact(artifact)
            (folder / "THREE_ARM_CHILD_RESULT.json").write_text(json.dumps(
                {"status": "success", "artifact": str(artifact)}), encoding="utf-8")
        else:
            (folder / "THREE_ARM_CHILD_RESULT.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8")
    return prior


def _registration(labels):
    return {"children": [{"family": "lsr_transform", "task": "t1", "seed": 2401, "arm": arm}
                        for arm in ("C", "D", "R")]}


def test_inherit_from_reuses_successful_children_and_runs_the_rest(tmp_path, monkeypatch):
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful={"t1__2401__C", "t1__2401__D"})
    registration = _registration(labels)

    calls = []
    def fake_run(reg, child, output):
        calls.append(child["arm"])
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = output / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / "art.json"
        _make_artifact(artifact)
        return {**child, "status": "generated", "output_dir": str(folder),
                "artifact": str(artifact), "bank_rows": [{"expression": "x0", "source": "core"}],
                "engine_usage": {"registered_engines_realized": True}}
    monkeypatch.setattr(runner, "run_child", fake_run)

    output = tmp_path / "out"
    rows = runner.resolve_execution_plan(registration, None, prior, output)

    inherited = [r for r in rows if r.get("inherited_from_v2")]
    run = [r for r in rows if not r.get("inherited_from_v2")]
    assert len(inherited) == 2 and len(run) == 1
    assert run[0]["arm"] == "R"  # the only non-successful prior child is run
    assert all(r["status"] == "generated" for r in rows)
    # Each inherited row records provenance without re-calling the provider.
    for r in inherited:
        assert r["inherited_source_execution"] == str(prior.resolve())
        assert r["inherited_source_result_sha256"]
    # The inherited artifact is copied locally, not referenced across experiments.
    assert (output / "children" / "t1__2401__C" / "art.json").is_file()


def test_child_list_requires_inherit_from_to_fill_matrix(tmp_path):
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful=set(labels))
    registration = _registration(labels)
    (tmp_path / "ids.txt").write_text("t1__2401__R\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="--child-list requires --inherit-from"):
        runner.resolve_execution_plan(
            registration, tmp_path / "ids.txt", None, output)


def test_child_list_runs_only_listed_and_inherits_rest(tmp_path, monkeypatch):
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful=set(labels))
    registration = _registration(labels)
    (tmp_path / "ids.txt").write_text("t1__2401__R\n", encoding="utf-8")
    calls = []
    def fake_run(reg, child, output):
        calls.append(child["arm"])
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = output / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / "art.json"
        _make_artifact(artifact)
        return {**child, "status": "generated", "output_dir": str(folder),
                "artifact": str(artifact), "bank_rows": [{"expression": "x0", "source": "core"}],
                "engine_usage": {"registered_engines_realized": True}}
    monkeypatch.setattr(runner, "run_child", fake_run)
    output = tmp_path / "out"
    rows = runner.resolve_execution_plan(
        registration, tmp_path / "ids.txt", prior, output)
    assert calls == ["R"]
    assert sum(r.get("inherited_from_v2") is True for r in rows) == 2


def test_inherit_fails_closed_when_child_list_target_has_no_inheritable_source(tmp_path):
    # Plan A fail-closed: a child listed to RUN is fine, but its non-listed peers
    # MUST be inheritable; if a peer failed in the prior run, the matrix is incomplete.
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful={"t1__2401__C"})  # D/R failed
    registration = _registration(labels)
    (tmp_path / "ids.txt").write_text("t1__2401__C\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        runner.resolve_execution_plan(
            registration, tmp_path / "ids.txt", prior, output)


def test_inherit_from_reruns_prior_failures_without_child_list(tmp_path, monkeypatch):
    # Default Plan A: inherit the prior successes, re-run the prior failures.
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful={"t1__2401__C"})  # D/R failed
    registration = _registration(labels)
    calls = []
    def fake_run(reg, child, output):
        calls.append(child["arm"])
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = output / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / "art.json"
        _make_artifact(artifact)
        return {**child, "status": "generated", "output_dir": str(folder),
                "artifact": str(artifact), "bank_rows": [{"expression": "x0", "source": "core"}],
                "engine_usage": {"registered_engines_realized": True}}
    monkeypatch.setattr(runner, "run_child", fake_run)
    output = tmp_path / "out"
    rows = runner.resolve_execution_plan(registration, None, prior, output)
    assert set(calls) == {"D", "R"}  # only the prior failures are re-run
    inherited = [r for r in rows if r.get("inherited_from_v2")]
    assert [r["arm"] for r in inherited] == ["C"]


def test_full_rerun_backward_compatible_without_flags(tmp_path, monkeypatch):
    labels = ["t1__2401__C", "t1__2401__D", "t1__2401__R"]
    prior = _make_prior_execution(tmp_path, labels, successful=set(labels))
    registration = _registration(labels)
    calls = []
    def fake_run(reg, child, output):
        calls.append(child["arm"])
        label = f"{child['task']}__{child['seed']}__{child['arm']}"
        folder = output / "children" / label
        folder.mkdir(parents=True, exist_ok=True)
        artifact = folder / "art.json"
        _make_artifact(artifact)
        return {**child, "status": "generated", "output_dir": str(folder),
                "artifact": str(artifact), "bank_rows": [{"expression": "x0", "source": "core"}],
                "engine_usage": {"registered_engines_realized": True}}
    monkeypatch.setattr(runner, "run_child", fake_run)
    output = tmp_path / "out"
    rows = runner.resolve_execution_plan(registration, None, None, output)
    assert calls == ["C", "D", "R"]
    assert all(not r.get("inherited_from_v2") for r in rows)
