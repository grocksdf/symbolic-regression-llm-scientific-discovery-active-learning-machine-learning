"""No-download source and split contract for Yacht confirmation."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "scientific-yacht-confirmation-source-registration-v1"


def run_yacht_source_candidate_gate(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "dataset_id", "official_url", "official_doi",
        "license", "download_md5", "published_instances",
        "published_features", "missing_values", "observation_type",
        "columns", "hull_group_columns", "speed_column", "target",
        "group_split", "grammar_contract", "github_evidence",
        "execution_authorized", "claim_boundary"}
    if (set(registration) != required
            or registration.get("schema") != SCHEMA
            or registration.get("execution_authorized") is not False):
        raise ValueError("invalid Yacht source registration")
    columns = list(registration["columns"])
    group_columns = list(registration["hull_group_columns"])
    split = registration["group_split"]
    group_total = sum(int(value) for value in split.values())
    checks = {
        "official_uci_identity_registered": (
            registration["dataset_id"] == "uci_yacht_hydrodynamics"
            and registration["official_doi"] == "10.24432/C5XG7R"),
        "official_license_registered":
            registration["license"] == "CC-BY-4.0",
        "fixed_download_checksum_registered": (
            len(registration["download_md5"]) == 32),
        "published_shape_registered": (
            registration["published_instances"] == 308
            and registration["published_features"] == 6),
        "published_missingness_closed":
            registration["missing_values"] is False,
        "experimental_measurement_source": (
            registration["observation_type"]
            == "Delft systematic yacht hull series resistance experiment"),
        "seven_columns_exactly_registered": (
            len(columns) == 7 and len(set(columns)) == 7),
        "hull_group_identity_uses_geometry_only": (
            len(group_columns) == 5
            and all(value in columns for value in group_columns)
            and registration["speed_column"] not in group_columns
            and registration["target"] not in group_columns),
        "speed_and_target_registered": (
            registration["speed_column"] == "froude_number"
            and registration["target"] == "residuary_resistance"),
        "group_split_is_complete_and_disjoint_by_construction": (
            set(split) == {
                "development", "validation", "acquisition_pool",
                "unused_open", "reserved_confirmation"}
            and group_total == 22
            and all(type(value) is int and value > 0
                    for value in split.values())),
        "row_arithmetic_matches_published_design": (
            group_total * 14 == registration["published_instances"]),
        "frozen_closed_basis_compatible":
            registration["grammar_contract"] == "pcpi-closed-basis-v1",
        "independent_pinned_github_evidence": (
            len(registration["github_evidence"]) >= 3
            and all(len(row.get("commit", "")) == 40
                    for row in registration["github_evidence"])),
        "execution_remains_blocked":
            registration["execution_authorized"] is False,
    }
    return {
        "schema": "scientific-yacht-confirmation-source-candidate-gate-v1",
        "checks": checks, "passed": all(checks.values()),
        "dataset_id": registration["dataset_id"],
        "group_split": dict(split),
        "expected_open_rows": (
            14 * sum(split[name] for name in (
                "development", "validation", "acquisition_pool"))),
        "reserved_confirmation_rows_expected":
            14 * split["reserved_confirmation"],
        "download_performed": False,
        "real_data_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "next_gate": "official-download-byte-and-schema-identity",
    }


__all__ = ["SCHEMA", "run_yacht_source_candidate_gate"]
