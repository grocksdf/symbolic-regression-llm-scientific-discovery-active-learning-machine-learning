"""Synthetic correctness fixtures only; real parser/compiler and fake HTTP."""
import json
from dataclasses import replace

import numpy as np
import pytest

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from hypothesis_mvp.discovery.predictive_bank_expansion import ExpansionDesign
from hypothesis_mvp.discovery.proposal_runtime import ProviderRoute, ProviderSettings
from scripts.bank_guided_llm import (
    CONTRACT, MODES, MULTISCALE_CONTRACT, MULTISCALE_MODES,
    LLMCompositionBudget, run_llm_comparison,
)


def fixture():
    ids = {name: tuple(f"{name}:{i}" for i in range(10))
           for name in ("fit", "calibration", "admission", "reporting")}
    design = ExpansionDesign(ids["fit"], ids["calibration"], ids["admission"], ids["reporting"],
                             (0,) * 10, (0,), tested_mixture_count=3)
    def data(offset, role=DataRole.VALIDATION):
        x = np.linspace(-.9, 1., 10)[:, None] + offset
        return RoleDataset(role, x, x[:, 0] ** 2 + .1 * x[:, 0])
    fit = data(0., DataRole.DEVELOPMENT)
    roles = {name: data(offset) for name, offset in
             (("calibration", .01), ("admission", .02), ("reporting", .03))}
    provider = ProviderSettings(routes=(ProviderRoute("https://fixture.invalid/v1", "fixture-model", "secret-fixture-key"),),
                                attempts=1, min_request_interval_s=0.)
    core = [{"expression": "x0", "source": "engine:fixture", "lineage_id": "frozen-core"}]
    return design, fit, roles, provider, core


def fake_http(monkeypatch, payloads, *, expression="x0**2", http_status=200, usage=True, malformed=False):
    def post(url, *, headers, json, timeout):
        import json as js
        payload = js.loads(json["messages"][1]["content"])
        payloads.append(payload)
        parent = payload["parent_hash"]
        content = {"protocol_id": "hypothesis-proposal-v1", "runtime_id": "canonical-real-only-discovery",
                   "round_id": payload["round_id"], "island": payload["island"],
                   "candidates": [{"candidate_id": "candidate", "parent_hash": parent,
                       "equation": expression, "action": "PROPOSE_NEW_SKELETON",
                       "rationale": "Synthetic contract fixture, not an efficacy claim."}]}
        class Response:
            status_code = http_status
            text = "secret-fixture-key should never enter audit"
            def json(self):
                return {"choices": [{"message": {"content": "bad-json" if malformed else js.dumps(content)}}],
                        **({"usage": {"prompt_tokens": 101, "completion_tokens": 17, "total_tokens": 118}}
                           if usage else {})}
        return Response()
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", post)


def execute(tmp_path, monkeypatch, *, reporting_offset=0., calls=1,
            implementation_contract=CONTRACT, diagnostic_axes=None, **fake_options):
    design, fit, roles, provider, core = fixture()
    modes = MULTISCALE_MODES if implementation_contract == MULTISCALE_CONTRACT else MODES
    if implementation_contract == MULTISCALE_CONTRACT:
        design = replace(design, tested_mixture_count=len(modes))
    root = tmp_path / "run"
    payloads, events = [], []
    fake_http(monkeypatch, payloads, **fake_options)
    def reader(name):
        def read():
            required = {"calibration": "GENERATION_FROZEN.json", "admission": "CALIBRATION_FROZEN.json",
                        "reporting": "ADMISSION_FROZEN.json"}
            assert (root / required[name]).exists()
            assert all((root / mode / "GENERATION.json").exists() for mode in modes)
            events.append(name)
            data = roles[name]
            if name == "reporting":
                data = RoleDataset(data.role, data.X, data.y + reporting_offset)
            return getattr(design, name + "_ids"), data
        return read
    result = run_llm_comparison(
        implementation_contract=implementation_contract, budget=LLMCompositionBudget(calls, 1, 1),
        design=design, core=core, fit=fit, provider_settings=provider,
        read_calibration=reader("calibration"), read_admission=reader("admission"),
        read_reporting=reader("reporting"), output_root=root, seed=12, diagnostic_axes=diagnostic_axes)
    return result, root, payloads, events


def test_real_llm_protocol_to_frozen_independent_admission(monkeypatch, tmp_path):
    result, root, payloads, events = execute(tmp_path, monkeypatch)
    assert result["protocol_complete"] and result["statistical_core_changed"] is False
    assert events == ["calibration", "admission", "reporting"]
    assert result["candidate_counts"] == dict.fromkeys(("bank", "residual", "blind"), 1)
    assert len(payloads) == 3
    rights = [{key: p["contract"][key] for key in
               ("max_candidates", "allowed_actions", "new_skeleton_quota", "allowed_symbols")}
              for p in payloads]
    assert rights[0] == rights[1] == rights[2]
    contexts = [p["scientific_context"]["bank_predictive_search_feedback"] for p in payloads]
    assert [p["mode"] for p in contexts] == ["bank", "residual", "blind"]
    assert contexts[2]["regions"] == []
    assert all(p["verified_structure_library"] == [] and p["accepted_refinements"] == [] for p in payloads)
    assert all(c["provider_attempts"] == 1 and c["logical_calls"] == 1
               and c["prompt_tokens"] == 101 for c in result["cost"].values())
    assert all(r["reporting_used_for_selection"] is False for r in result["results"].values())
    for path in root.rglob("*.json"):
        assert "secret-fixture-key" not in path.read_text()


def test_reporting_cannot_change_prompts_candidates_or_decisions(monkeypatch, tmp_path):
    _, a, first, _ = execute(tmp_path / "first", monkeypatch)
    _, b, second, _ = execute(tmp_path / "second", monkeypatch, reporting_offset=100.)
    assert first == second
    for name in ("GENERATION_FROZEN.json", "COMPONENTS_FROZEN.json", "CALIBRATION_FROZEN.json", "ADMISSION_FROZEN.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_duplicate_requests_count_and_do_not_duplicate_bank(monkeypatch, tmp_path):
    result, _, payloads, _ = execute(tmp_path, monkeypatch, calls=2)
    assert len(payloads) == 6
    assert set(result["candidate_counts"].values()) == {1}
    assert all(c["logical_calls"] == 2 and c["provider_attempts"] == 2 for c in result["cost"].values())


def test_unknown_tokens_are_unknown_not_zero(monkeypatch, tmp_path):
    result, _, _, _ = execute(tmp_path, monkeypatch, usage=False)
    assert all(c["prompt_tokens"] is None and c["completion_tokens"] is None for c in result["cost"].values())


def test_schema_failure_is_recorded_without_extra_calls_or_zero_score(monkeypatch, tmp_path):
    result, root, payloads, _ = execute(tmp_path, monkeypatch, malformed=True)
    assert len(payloads) == 3
    assert set(result["candidate_counts"].values()) == {0}
    for mode in ("bank", "residual", "blind"):
        record = json.loads((root / mode / "CALL_0000.json").read_text())
        assert record["protocol_valid"] is False and record["rejected_candidates"]
        assert "search_gain" not in record
        assert result["results"][mode]["gains"]["screened_predictive_stacking"] == 0.


def test_provider_failure_never_opens_independent_roles(monkeypatch, tmp_path):
    design, fit, _, provider, core = fixture()
    payloads = []
    fake_http(monkeypatch, payloads, http_status=429)
    def forbidden():
        pytest.fail("independent responses opened after provider failure")
    root = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="provider/protocol failed"):
        run_llm_comparison(implementation_contract=CONTRACT, budget=LLMCompositionBudget(2, 1, 1),
            design=design, core=core, fit=fit, provider_settings=provider,
            read_calibration=forbidden, read_admission=forbidden, read_reporting=forbidden,
            output_root=root, seed=12)
    assert len(payloads) == 1 and not (root / "GENERATION_FROZEN.json").exists()
    assert (root / "FAILURE.json").exists()
    cost = json.loads((root / "bank" / "FAILURE.json").read_text())["cost"]
    assert cost["provider_failures"] == 1 and cost["transport_records"][0]["http_status"] == 429
    assert all("secret-fixture-key" not in p.read_text() for p in root.rglob("*.json"))


def test_undefined_or_duplicate_candidate_cannot_receive_predictive_mass(monkeypatch, tmp_path):
    result, root, _, _ = execute(tmp_path / "undefined", monkeypatch, expression="log(x0)")
    assert set(result["candidate_counts"].values()) == {0}
    assert json.loads((root / "bank" / "CALL_0000.json").read_text())["candidates"][0]["status"] == "invalid-on-fit-domain"
    duplicate, _, _, _ = execute(tmp_path / "duplicate", monkeypatch, expression="x0")
    assert set(duplicate["candidate_counts"].values()) == {0}


@pytest.mark.parametrize("change", ["legacy", "heldout", "multiplicity", "endpoint-secret"])
def test_preflight_failure_happens_before_network_or_output(monkeypatch, tmp_path, change):
    design, fit, _, provider, core = fixture()
    contract = CONTRACT
    if change == "legacy":
        contract = "bank-location-guided-search-v1"
    if change == "heldout":
        fit = RoleDataset(DataRole.UNTOUCHED_HELDOUT, fit.X, fit.y)
    if change == "multiplicity":
        design = replace(design, tested_mixture_count=1)
    if change == "endpoint-secret":
        provider = replace(provider, routes=(replace(provider.routes[0], base_url="https://user:password@example.invalid"),))
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post",
                        lambda *a, **k: pytest.fail("network opened"))
    with pytest.raises(ValueError):
        run_llm_comparison(implementation_contract=contract, budget=LLMCompositionBudget(1, 1, 1),
            design=design, core=core, fit=fit, provider_settings=provider,
            read_calibration=None, read_admission=None, read_reporting=None, output_root=tmp_path / "bad", seed=12)
    assert not (tmp_path / "bad").exists()


def test_overlap_before_reporting_is_terminal(monkeypatch, tmp_path):
    design, fit, _, provider, core = fixture()
    fake_http(monkeypatch, [])
    with pytest.raises(ValueError, match="overlap"):
        run_llm_comparison(implementation_contract=CONTRACT, budget=LLMCompositionBudget(1, 1, 1),
            design=design, core=core, fit=fit, provider_settings=provider,
            read_calibration=lambda: (design.calibration_ids, RoleDataset(DataRole.VALIDATION, fit.X, fit.y)),
            read_admission=lambda: pytest.fail("admission opened"),
            read_reporting=lambda: pytest.fail("report opened"), output_root=tmp_path / "overlap", seed=12)
    assert (tmp_path / "overlap" / "GENERATION_FROZEN.json").exists()
    assert not (tmp_path / "overlap" / "CALIBRATION_FROZEN.json").exists()


def test_multiscale_real_pipeline_has_exact_views_and_common_rights(monkeypatch, tmp_path):
    result, root, payloads, events = execute(
        tmp_path, monkeypatch, implementation_contract=MULTISCALE_CONTRACT, diagnostic_axes=(0,))
    assert events == ["calibration", "admission", "reporting"]
    assert result["protocol_complete"] and result["schema"] == MULTISCALE_CONTRACT
    assert result["candidate_counts"] == dict.fromkeys(MULTISCALE_MODES, 1)
    assert len(payloads) == 4
    plan = json.loads((root / "PLAN.json").read_text())
    assert plan["diagnostic_axes"] == [0] and plan["arm_execution_order"] == list(MULTISCALE_MODES)
    assert plan["design"]["tested_mixture_count"] == 4
    contexts = [p["scientific_context"]["bank_predictive_search_feedback"] for p in payloads]
    blind, global_only, local_only, joint = contexts
    assert [p["diagnostic_scope"] for p in contexts] == ["none", "global", "local", "joint"]
    assert all(p["schema"] == "bank-multiscale-proposal-feedback-v1" for p in contexts)
    assert blind["regions"] == global_only["regions"] == []
    assert blind["global_summary"] is local_only["global_summary"] is None
    assert global_only["global_summary"] == joint["global_summary"]
    assert local_only["regions"] == joint["regions"]
    assert all(p["global_local_links"] == [] for p in contexts[:3])
    assert len(joint["global_local_links"]) == len(joint["regions"]) == 2
    for region, link in zip(joint["regions"], joint["global_local_links"]):
        assert link["location_score_minus_global"] == (
            region["mean_location_score"] - joint["global_summary"]["mean_location_score"])
    # These views reuse one snapshot, rather than fitting a different bank per arm.
    assert all(p["bank_inventory"] == blind["bank_inventory"]
               and p["bank_inventory_identity"] == blind["bank_inventory_identity"] for p in contexts)
    rights = [{key: p["contract"][key] for key in
               ("max_candidates", "allowed_actions", "new_skeleton_quota", "allowed_symbols")}
              for p in payloads]
    assert all(r == rights[0] for r in rights)
    assert len({p["parent_hash"] for p in payloads}) == 1
    assert all(p["verified_structure_library"] == [] and p["accepted_refinements"] == [] for p in payloads)
    assert all(c["logical_calls"] == c["provider_attempts"] == 1 for c in result["cost"].values())
    for mode, scope in plan["diagnostic_scopes"].items():
        generation = json.loads((root / mode / "GENERATION.json").read_text())
        assert generation["schema"] == MULTISCALE_CONTRACT and generation["diagnostic_scope"] == scope
    # Weighted regional means already determine the global summaries: this is
    # an explicit information-presentation ablation, not independent evidence.
    summary = joint["global_summary"]
    for metric in set(summary) - {"count"}:
        assert sum(r[metric] * r["count"] for r in joint["regions"]) / summary["count"] == pytest.approx(summary[metric])


def test_multiscale_reporting_labels_cannot_change_any_generation_or_admission(monkeypatch, tmp_path):
    options = {"implementation_contract": MULTISCALE_CONTRACT, "diagnostic_axes": (0,)}
    _, a, first, _ = execute(tmp_path / "first", monkeypatch, **options)
    _, b, second, _ = execute(tmp_path / "second", monkeypatch, reporting_offset=100., **options)
    assert first == second
    for name in ("GENERATION_FROZEN.json", "COMPONENTS_FROZEN.json", "CALIBRATION_FROZEN.json", "ADMISSION_FROZEN.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_legacy_contract_remains_three_arms_and_original_prompt_schema(monkeypatch, tmp_path):
    result, root, payloads, _ = execute(tmp_path, monkeypatch)
    assert result["schema"] == CONTRACT and tuple(result["results"]) == MODES
    assert len(payloads) == 3
    assert "diagnostic_axes" not in json.loads((root / "PLAN.json").read_text())
    assert all("diagnostic_scope" not in p["scientific_context"]["bank_predictive_search_feedback"] for p in payloads)


@pytest.mark.parametrize("axes", [None, (), [0], (0, 0), (-1,), (1,), (True,), (0.0,), ("0",)])
def test_multiscale_bad_axes_fail_before_output_or_transport(monkeypatch, tmp_path, axes):
    design, fit, _, provider, core = fixture()
    design = replace(design, tested_mixture_count=4)
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post",
                        lambda *a, **k: pytest.fail("transport opened"))
    with pytest.raises(ValueError, match="axes"):
        run_llm_comparison(implementation_contract=MULTISCALE_CONTRACT,
            diagnostic_axes=axes, budget=LLMCompositionBudget(1, 1, 1),
            design=design, core=core, fit=fit, provider_settings=provider,
            read_calibration=None, read_admission=None, read_reporting=None,
            output_root=tmp_path / "invalid", seed=12)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("contract,axes,count", [(CONTRACT, (0,), 4), (MULTISCALE_CONTRACT, (0,), 3)])
def test_multiscale_requires_explicit_contract_and_four_mixture_policy(monkeypatch, tmp_path, contract, axes, count):
    design, fit, _, provider, core = fixture()
    design = replace(design, tested_mixture_count=count)
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post",
                        lambda *a, **k: pytest.fail("transport opened"))
    with pytest.raises(ValueError):
        run_llm_comparison(implementation_contract=contract, diagnostic_axes=axes,
            budget=LLMCompositionBudget(1, 1, 1), design=design, core=core,
            fit=fit, provider_settings=provider, read_calibration=None,
            read_admission=None, read_reporting=None, output_root=tmp_path / "invalid", seed=12)
    assert not (tmp_path / "invalid").exists()
