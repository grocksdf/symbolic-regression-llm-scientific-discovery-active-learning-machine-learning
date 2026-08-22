"""Response-free CERT.22 operational identity and scale NO-GO checks."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hypothesis_mvp.data.real_registry import REAL_DATASET_SPECS
from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT22_HELDOUT_ACCESS_AUTHORIZED,
    P3F4_CERT22_OPERATIONAL_EXECUTION_AUTHORIZED,
    P3F4_CERT22_OPERATIONAL_H0_BINDING_AUTHORIZED,
    P3F4_CERT22_OUTPUT_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT22_REAL_DATA_ACCESS_AUTHORIZED,
    P3F4_CERT22_RESPONSE_FREE_PREFLIGHT_AUTHORIZED,
    P3F4_CERT22_SYSTEM_ENTROPY_ACCESS_AUTHORIZED,
    RegisteredH0FamilyPlan,
    RegisteredH0Task,
    build_cert22_no_go_preflight,
    cumulative_raw_ast_count,
    reachable_monomial_class_lower_bound,
)
from tests.test_pcpi_p3f4_cert20_exact_rejection_source import _source_fixture


ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads(
    (ROOT / "configs/p3f_4_cert22_operational_preflight_freeze.json").read_text(
        encoding="utf-8"
    )
)


def _file_hash(relative: str) -> str:
    return sha256((ROOT / relative).read_bytes()).hexdigest()


def _family() -> RegisteredH0FamilyPlan:
    config = json.loads((ROOT / FREEZE["registered_real_family"]["config"]).read_text())
    return RegisteredH0FamilyPlan(
        config_sha256=_file_hash(FREEZE["registered_real_family"]["config"]),
        seeds=tuple(config["seeds"]),
        tasks=tuple(
            RegisteredH0Task(
                item["dataset_id"],
                item["feature_count"],
                len(config["seeds"]),
                config["initial_observation_budget"],
            )
            for item in FREEZE["registered_real_family"]["datasets"]
        ),
        split_seed=config["split_seed"],
        initial_index_rule=FREEZE["registered_real_family"]["initial_index_rule"],
        standardizer_rule=FREEZE["registered_real_family"]["standardizer_rule"],
    )


def _preflight():
    inherited = FREEZE["inherited_target"]
    executable = FREEZE["inherited_executable_source_fixture"]
    return build_cert22_no_go_preflight(
        _family(),
        frozen_target_config_sha256=_file_hash(inherited["config"]),
        frozen_source_config_sha256=_file_hash(
            "configs/p3f_4_cert20_exact_rejection_source_freeze.json"
        ),
        frozen_runner_config_sha256=_file_hash(
            "configs/p3f_4_cert21_guarded_runner_freeze.json"
        ),
        frozen_target_feature_count=inherited["feature_count"],
        frozen_target_maximum_nodes=inherited["semantic_core_maximum_nodes"],
        executable_source_feature_count=executable["feature_count"],
        executable_source_maximum_nodes=executable["maximum_nodes"],
        component_count=inherited["component_count"],
    )


def test_cert22_authorizes_preflight_only() -> None:
    assert P3F4_CERT22_RESPONSE_FREE_PREFLIGHT_AUTHORIZED
    assert not P3F4_CERT22_OPERATIONAL_H0_BINDING_AUTHORIZED
    assert not P3F4_CERT22_OPERATIONAL_EXECUTION_AUTHORIZED
    assert not P3F4_CERT22_SYSTEM_ENTROPY_ACCESS_AUTHORIZED
    assert not P3F4_CERT22_OUTPUT_MATERIALIZATION_AUTHORIZED
    assert not P3F4_CERT22_REAL_DATA_ACCESS_AUTHORIZED
    assert not P3F4_CERT22_HELDOUT_ACCESS_AUTHORIZED


def test_registered_family_is_three_targets_eight_seeds_and_twenty_four_h0() -> None:
    family = _family()
    assert tuple(item.dataset_id for item in family.tasks) == (
        "uci_ccpp",
        "uci_gas_turbine_co",
        "uci_gas_turbine_nox",
    )
    assert family.required_h0_artifact_count == family.missing_h0_artifact_count == 24
    assert not family.bound_h0_artifact_hashes


def test_registered_feature_dimensions_match_registry_without_loading_data() -> None:
    family = _family()
    assert tuple(item.feature_count for item in family.tasks) == (4, 9, 9)
    assert all(
        item.feature_count == len(REAL_DATASET_SPECS[item.dataset_id].feature_names)
        for item in family.tasks
    )


def test_raw_ast_recurrence_matches_registered_one_dimensional_evidence() -> None:
    assert cumulative_raw_ast_count(1, 17) == 5_924_484_194
    assert cumulative_raw_ast_count(4, 17) == 5_480_405_422_085
    assert cumulative_raw_ast_count(9, 17) == 1_331_131_316_840_170


def test_monomial_lower_bound_is_analytic_and_dimension_aware() -> None:
    assert reachable_monomial_class_lower_bound(1, 17) == 10
    assert reachable_monomial_class_lower_bound(4, 17) == 715
    assert reachable_monomial_class_lower_bound(9, 17) == 48_620


def test_target_ball_lower_bound_covers_all_registered_h0() -> None:
    preflight = _preflight()
    assert tuple(row.target_ball_lower_bound_family for row in preflight.scale_rows) == (
        17_160,
        1_166_880,
        1_166_880,
    )
    assert preflight.target_ball_lower_bound_family == 2_350_920


def test_cert20_executable_fixture_is_one_dimensional_cutoff_one_not_j17() -> None:
    provider, _, source, balls = _source_fixture()
    assert provider.target_contract.grammar.feature_count == 1
    assert source.maximum_nodes == 1
    assert len(balls) == 6
    assert FREEZE["inherited_target"]["semantic_core_maximum_nodes"] == 17


def test_dimension_cutoff_and_missing_h0_are_explicit_blockers() -> None:
    blockers = set(_preflight().blockers)
    assert "registered-real-dimensions-not-bound-by-frozen-target" in blockers
    assert "executable-source-fixture-does-not-bind-frozen-target-scale" in blockers
    assert "operational-h0-artifact-identities-absent" in blockers


def test_missing_time_and_storage_are_not_replaced_by_guesses() -> None:
    blockers = set(_preflight().blockers)
    assert "certified-wall-clock-bound-absent" in blockers
    assert "certified-storage-bound-absent" in blockers
    assert _preflight().status == "no-go"


def test_output_identity_is_unique_but_cannot_be_materialized() -> None:
    first = _preflight()
    second = _preflight()
    assert first.output_identity == second.output_identity
    assert first.output_publication == "reserved-identity-only-no-overwrite"
    assert not first.output_materialized
    changed = replace(_family(), split_seed=_family().split_seed + 1)
    changed_preflight = build_cert22_no_go_preflight(
        changed,
        frozen_target_config_sha256=first.frozen_target_config_sha256,
        frozen_source_config_sha256=first.frozen_source_config_sha256,
        frozen_runner_config_sha256=first.frozen_runner_config_sha256,
        frozen_target_feature_count=1,
        frozen_target_maximum_nodes=17,
        executable_source_feature_count=1,
        executable_source_maximum_nodes=1,
        component_count=3,
    )
    assert changed_preflight.output_identity != first.output_identity


def test_h0_hash_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_family(), bound_h0_artifact_hashes=("not-a-hash",))
    repeated = ("0" * 64,) * 2
    with pytest.raises(ValueError, match="identity is invalid"):
        replace(_family(), bound_h0_artifact_hashes=repeated)


def test_freeze_matches_exact_no_go_decision() -> None:
    preflight = _preflight()
    assert FREEZE["decision"]["status"] == preflight.status
    assert tuple(FREEZE["decision"]["blockers"]) == preflight.blockers
    assert (
        FREEZE["response_free_scale_lower_bounds"][
            "complete_target_balls_across_24_h0_minimum"
        ]
        == preflight.target_ball_lower_bound_family
    )
    assert FREEZE["authorization"]["formal_experiment"] is False
