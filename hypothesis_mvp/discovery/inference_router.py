"""Fail-closed routing between finite exact inference and certified open SMC."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any


@dataclass(frozen=True)
class InferencePlan:
    mode: str
    support_kind: str
    model_identity: str
    certified_smc_authorized: bool
    reason: str

    def __post_init__(self):
        if (self.mode not in {"exact_finite", "certified_open_smc"}
                or self.support_kind not in {"finite_frozen", "open_transdimensional"}
                or not self.model_identity or not self.reason
                or (self.mode == "certified_open_smc"
                    and not self.certified_smc_authorized)):
            raise ValueError("invalid inference routing plan")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "scientific-inference-routing-plan-v1",
            "mode": self.mode, "support_kind": self.support_kind,
            "model_identity": self.model_identity,
            "certified_smc_authorized": self.certified_smc_authorized,
            "reason": self.reason}

    @property
    def stable_hash(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()


def route_inference(model: Any, *, requested_mode: str = "auto",
                    open_support: bool = False,
                    certified_smc_authorized: bool = False) -> InferencePlan:
    if requested_mode not in {"auto", "exact_finite", "certified_open_smc"}:
        raise ValueError("unknown scientific inference mode")
    identity = str(getattr(model, "stable_hash", ""))
    if not open_support:
        if requested_mode == "certified_open_smc":
            raise ValueError("open SMC cannot target a finite frozen model")
        return InferencePlan(
            "exact_finite", "finite_frozen", identity, False,
            "finite frozen support admits exact conjugate inference")
    if requested_mode == "exact_finite":
        raise ValueError("exact finite inference cannot target open support")
    if not certified_smc_authorized:
        raise PermissionError(
            "open transdimensional inference requires certified SMC authorization")
    return InferencePlan(
        "certified_open_smc", "open_transdimensional", identity, True,
        "open support explicitly authorized by certified SMC integration gate")


__all__ = ["InferencePlan", "route_inference"]
