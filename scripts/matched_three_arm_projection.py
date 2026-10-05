"""Shared-admission projection for E, E+S-matched, and E+L-gap."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from hypothesis_mvp.data.roles import RoleDataset
from hypothesis_mvp.discovery.candidate_region_expansion import (
    FrozenAxisRegions,
)
from hypothesis_mvp.discovery.pcpi_adapter import EXPANDED_FORMULA_POLICY
from hypothesis_mvp.discovery.regional_candidate_admission import (
    admit_regional_candidates,
)
from hypothesis_mvp.pcpi.reference import NormalInverseGammaPrior


def _identity(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, default=str,
        separators=(",", ":")).encode()).hexdigest()


def project_matched_three_arms(
        *, engine_rows: Sequence[Mapping],
        matched_rows: Sequence[Mapping], llm_rows: Sequence[Mapping],
        fit: RoleDataset, gap_audit: RoleDataset,
        admission: RoleDataset, domain: np.ndarray,
        regions: FrozenAxisRegions, eligible_regions: tuple[int, ...],
        prior: NormalInverseGammaPrior, measurement_budget: int,
        alpha: float, proposal_cap: int = 6) -> dict:
    core = [dict(row) for row in engine_rows]
    matched = [dict(row) for row in matched_rows]
    llm = [dict(row) for row in llm_rows]
    if (not core or proposal_cap != 6 or len(matched) > proposal_cap
            or len(llm) > proposal_cap
            or any(row.get("origin") != "non_llm_matched"
                   for row in matched)
            or any(row.get("origin") != "llm"
                   for row in llm)):
        raise ValueError("matched three-arm candidate contract changed")
    shared = {
        "core_identity": _identity(core),
        "fit_identity": fit.fingerprint,
        "gap_identity": gap_audit.fingerprint,
        "admission_identity": admission.fingerprint,
        "domain_identity": sha256(np.ascontiguousarray(
            domain, dtype=np.float64).tobytes()).hexdigest(),
        "region_identity": regions.identity,
        "eligible_regions": list(eligible_regions),
        "measurement_budget": measurement_budget,
        "alpha": alpha,
        "proposal_cap": proposal_cap,
    }
    retained_s, audit_s = admit_regional_candidates(
        core, matched, fit, gap_audit, admission, domain,
        regions, eligible_regions, prior, measurement_budget,
        alpha=alpha, coefficient_policy=EXPANDED_FORMULA_POLICY)
    retained_l, audit_l = admit_regional_candidates(
        core, llm, fit, gap_audit, admission, domain,
        regions, eligible_regions, prior, measurement_budget,
        alpha=alpha, coefficient_policy=EXPANDED_FORMULA_POLICY)
    arms = {
        "E": core,
        "E_plus_S_gap_matched": [*core, *retained_s],
        "E_plus_L_gap": [*core, *retained_l],
    }
    if (arms["E_plus_S_gap_matched"][:len(core)] != core
            or arms["E_plus_L_gap"][:len(core)] != core):
        raise ValueError("optional arm changed the frozen engine core")
    return {
        "schema": "matched-three-arm-admission-projection-v1",
        "shared_contract": shared,
        "arms": arms,
        "admission_audits": {
            "E_plus_S_gap_matched": audit_s,
            "E_plus_L_gap": audit_l,
        },
        "attempt_counts": {
            "E_plus_S_gap_matched": len(matched),
            "E_plus_L_gap": len(llm),
        },
        "retained_counts": {
            "E_plus_S_gap_matched": len(retained_s),
            "E_plus_L_gap": len(retained_l),
        },
        "candidate_response_accessed": False,
        "heldout_opened": False,
    }


__all__ = ["project_matched_three_arms"]
