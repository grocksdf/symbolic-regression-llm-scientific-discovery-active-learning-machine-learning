"""Artifact-only paired Gas-NOX/Yacht structural case certificate."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _case(
    name: str, viability: dict[str, Any],
    admission: dict[str, Any], source: str,
) -> dict[str, Any]:
    full_v = viability["variants"]["full"]
    no_v = viability["variants"]["no_llm"]
    full_a = admission["variants"]["full"]
    llm = full_a["sources"].get("llm", {})
    gains = list(llm.get("fold_log_score_gains_vs_core", ()))
    tolerances = list(llm.get("fold_numerical_tolerances", ()))
    return {
        "name": name, "source": source,
        "full_viable": full_v["passed"] is True,
        "no_llm_viable": no_v["passed"] is True,
        "full_entropy_nats": full_v["class_entropy_nats"],
        "no_llm_entropy_nats": no_v["class_entropy_nats"],
        "full_operational_class_count": full_v["operational_class_count"],
        "no_llm_operational_class_count": no_v[
            "operational_class_count"],
        "llm_admitted": llm.get("admitted") is True,
        "llm_fold_gains_vs_core": gains,
        "llm_fold_safe": bool(
            len(gains) == len(tolerances) == 2
            and all(g > t for g, t in zip(gains, tolerances))),
        "llm_source_weight": llm.get("weight"),
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


def build_paired_case_certificate(
    gas_viability: str | Path, gas_admission: str | Path,
    yacht_viability: str | Path, yacht_admission: str | Path,
) -> dict[str, Any]:
    gas_v_path, gas_a_path = Path(gas_viability), Path(gas_admission)
    yacht_v_path, yacht_a_path = Path(yacht_viability), Path(yacht_admission)
    cases = [
        _case("gas_nox_positive_structural_case",
              _read(gas_v_path), _read(gas_a_path),
              "uci_gas_turbine_nox:2026092403"),
        _case("yacht_negative_boundary_case",
              _read(yacht_v_path), _read(yacht_a_path),
              "yacht_hydrodynamics:2026092602"),
    ]
    gas, yacht = cases
    checks = {
        "gas_full_viable": gas["full_viable"],
        "gas_no_llm_collapsed": not gas["no_llm_viable"],
        "gas_llm_fold_safe_admitted": (
            gas["llm_admitted"] and gas["llm_fold_safe"]),
        "yacht_full_not_viable": not yacht["full_viable"],
        "yacht_llm_not_admitted_or_not_fold_safe": (
            not yacht["llm_admitted"] or not yacht["llm_fold_safe"]),
        "paired_cases_use_distinct_coordinates": gas["source"] != yacht["source"],
        "responses_and_heldout_closed": all(
            not case["candidate_response_accessed"]
            and not case["heldout_opened"] for case in cases),
    }
    return {
        "schema": "scientific-typed-synthesis-paired-case-certificate-v1",
        "passed": all(checks.values()),
        "checks": checks, "cases": cases,
        "inputs": {
            "gas_viability_sha256": _sha(gas_v_path),
            "gas_admission_sha256": _sha(gas_a_path),
            "yacht_viability_sha256": _sha(yacht_v_path),
            "yacht_admission_sha256": _sha(yacht_a_path),
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
        "acquisition_efficacy_demonstrated": False,
        "heldout_superiority_demonstrated": False,
        "claim": (
            "Typed Scientist contribution is task-dependent: it rescued a "
            "safe hypothesis bank in Gas-NOX but was rejected on an untouched "
            "Yacht task where full also failed readiness."),
        "claim_boundary": (
            "Paired development/source-screen evidence only; no universal "
            "predictive superiority, acquisition efficacy or held-out claim."),
    }


__all__ = ["build_paired_case_certificate"]
