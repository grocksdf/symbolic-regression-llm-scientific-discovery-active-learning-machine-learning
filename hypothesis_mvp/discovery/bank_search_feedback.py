"""Search-only predictive-location feedback from a complete frozen Bank.

For p(y|x)=sum_k pi_k t_k(y;mu_k,s_k,nu_k), the direction for a common
location shift is d log p(y-delta|x)/d delta at zero. This local search
surrogate uses the whole predictive distribution, not the best formula's MSE.
It is NOT the derivative for adding an arbitrary symbolic component, an
admission statistic, a calibration guarantee or an originality claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from hypothesis_mvp.data.roles import DataRole
from .proposal_calibration import assert_generation_context_safe


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             default=lambda v: v.tolist()).encode()).hexdigest()


def mixture_location_feedback(locations, scales, degrees, log_weights, targets):
    """Analytic location score and ordinary MAP-component residual.

    All matrices use rows x components. Degrees and weights use components.
    Zero mass is allowed (-inf log weight); invalid scales remain failures.
    No clipping/epsilon changes the fitted predictive distribution.
    """
    mu, scale = (np.asarray(v, dtype=float) for v in (locations, scales))
    nu, logw, y = (np.asarray(v, dtype=float) for v in (degrees, log_weights, targets))
    if (mu.ndim != 2 or not len(mu) or mu.shape[1] == 0 or scale.shape != mu.shape
            or nu.shape != (mu.shape[1],) or logw.shape != nu.shape or y.shape != (len(mu),)
            or any(not np.all(np.isfinite(v)) for v in (mu, scale, nu, y))
            or np.any(scale <= 0) or np.any(nu <= 0) or np.any(np.isnan(logw))
            or np.any(logw == np.inf) or not np.any(np.isfinite(logw))
            or abs(float(logsumexp(logw))) > 1e-10):
        raise ValueError("invalid normalized Student-t predictive Bank")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        residual = y[:, None] - mu
        log_components = student_t.logpdf(y[:, None], df=nu, loc=mu, scale=scale)
        joint = log_components + logw
        log_density = logsumexp(joint, axis=1)
        responsibility = np.exp(joint - log_density[:, None])
        # Branches avoid squaring a very large standardized residual.
        z = residual / scale
        ratio = np.empty_like(z)
        small = np.abs(z) <= 1
        broadcast_nu = np.broadcast_to(nu, z.shape)
        ratio[small] = z[small] / (broadcast_nu[small] + z[small] ** 2)
        inv = 1. / z[~small]
        ratio[~small] = inv / (1. + broadcast_nu[~small] * inv ** 2)
        component_score = (nu + 1.) * ratio / scale
        score = np.sum(responsibility * component_score, axis=1)
        map_residual = residual[:, int(np.argmax(logw))]
    if not all(np.all(np.isfinite(v)) for v in (log_density, responsibility, score, map_residual)):
        raise FloatingPointError("nonfinite predictive Bank feedback")
    return log_density, score, map_residual


@dataclass(frozen=True)
class BankSearchFeedback:
    bank_identity: str
    search_row_ids: tuple[str, ...]
    search_data_identity: str
    log_density: tuple[float, ...]
    location_score: tuple[float, ...]
    residual: tuple[float, ...]
    regions_json: str

    @property
    def identity(self):
        return _hash(self.__dict__)

    def direction(self, mode):
        if mode == "bank":
            return np.asarray(self.location_score)
        if mode == "residual":
            return np.asarray(self.residual)
        if mode == "blind":
            return np.zeros(len(self.search_row_ids))
        raise ValueError("unknown search feedback mode")

    def prompt_payload(self, mode="bank"):
        if mode not in {"bank", "residual", "blind"}:
            raise ValueError("unknown search feedback mode")
        fields = {"bank": ("mean_log_density", "mean_location_score"),
                  "residual": ("mean_residual",), "blind": ()}[mode]
        regions = [{key: value for key, value in row.items()
                    if key in ("axis", "side", "cut", "count", *fields)}
                   for row in json.loads(self.regions_json)] if fields else []
        return {"schema": "bank-search-feedback-v1", "mode": mode,
                "signal_identity": self.identity if mode != "blind" else None,
                "source_role": "opened-generation-search-only", "regions": regions,
                "instruction": "Use this local diagnostic to propose structural changes. "
                               "Positive mean_location_score favors a small upward shift of "
                               "the Bank predictive locations; negative favors downward. "
                               "A residual is observed target minus fitted MAP prediction. "
                               "These are not gradients of arbitrary AST replacements. "
                               "Preserve registered grammar and search rights; the signal "
                               "does not select or certify the final predictor."}


def feedback_from_posterior(engine, posterior, search, *, search_row_ids,
                            forbidden_row_ids, bank_identity):
    """Read only opened generation/search roles; require explicit row exclusion.

    forbidden_row_ids are opaque registered admission/reporting indices, not
    response hashes. The caller must bind these IDs to actual dataset rows.
    """
    ids, forbidden = tuple(search_row_ids), tuple(forbidden_row_ids)
    if (search.role not in {DataRole.DEVELOPMENT, DataRole.VALIDATION}
            or len(ids) != len(search.X) or len(set(ids)) != len(ids)
            or not forbidden or any(not isinstance(v, str) or not v for v in (*ids, *forbidden))
            or set(ids).intersection(forbidden) or not isinstance(bank_identity, str)
            or not bank_identity or posterior.bank_hash != engine.bank.stable_hash):
        raise ValueError("feedback needs a bound Bank and isolated search rows")
    locations, scales, degrees = [], [], []
    for member in posterior.members:
        rows = engine.design_rows(search.X, member.structure)
        params = engine.conditional_parameters(member)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            variance_factor = 1. + np.einsum("ij,jk,ik->i", rows, params.covariance_factor, rows)
            locations.append(rows @ params.mean)
            scales.append(np.sqrt(params.noise_scale / params.noise_shape * variance_factor))
            degrees.append(2. * params.noise_shape)
    lpd, score, residual = mixture_location_feedback(
        np.column_stack(locations), np.column_stack(scales), degrees,
        posterior.log_probabilities, search.y)
    regions = []
    for axis in range(search.X.shape[1]):
        cut = float(np.median(search.X[:, axis]))
        for side, mask in (("le", search.X[:, axis] <= cut), ("gt", search.X[:, axis] > cut)):
            if not np.any(mask):
                continue
            regions.append({"axis": axis, "side": side, "cut": cut, "count": int(np.sum(mask)),
                            "mean_log_density": float(np.mean(lpd[mask])),
                            "mean_location_score": float(np.mean(score[mask])),
                            "mean_residual": float(np.mean(residual[mask]))})
    return BankSearchFeedback(bank_identity, ids, search.fingerprint,
                              tuple(map(float, lpd)), tuple(map(float, score)),
                              tuple(map(float, residual)), json.dumps(regions, allow_nan=False))


def rank_materials(feedback, material_values, *, mode):
    """Unit-L2 directional alignment, with signed evidence and no response fit.

    Materials are centered; constant location shifts are already refit by the
    existing coefficient model. Absolute alignment allows either coefficient
    sign. The score is a proposal heuristic; full candidates still need scoring.
    """
    direction = feedback.direction(mode)
    direction_scale = float(np.max(np.abs(direction)))
    unit_direction = direction / direction_scale if direction_scale else np.zeros_like(direction)
    direction_norm = float(np.linalg.norm(unit_direction))
    records = []
    for expression, values in material_values.items():
        values = np.asarray(values, dtype=float)
        if values.shape != direction.shape or not np.all(np.isfinite(values)):
            raise ValueError("material predictions must align with the search rows")
        scale = float(np.max(np.abs(values)))
        centered = values / scale if scale else np.zeros_like(values)
        centered = centered - np.mean(centered)
        norm = float(np.linalg.norm(centered))
        alignment = float(np.dot(centered / norm, unit_direction / direction_norm)) if norm and direction_norm else 0.
        records.append({"expression": expression, "alignment": alignment,
                        "priority": abs(alignment), "constant": norm == 0.})
    return sorted(records, key=lambda row: (-row["priority"], row["expression"]))


def attach_feedback(context, feedback, *, mode):
    """Explicit LLM integration hook; no grammar/quota modification or transport."""
    assert_generation_context_safe(context)
    if (context.get("posterior_gap_brief") is not None
            or (context.get("scientist_orchestration") or {}).get("posterior_gap_brief") is not None):
        raise ValueError("historical Gap prompts cannot be mixed with Bank-location guidance")
    result = json.loads(json.dumps(context, allow_nan=False))
    result["bank_predictive_search_feedback"] = feedback.prompt_payload(mode)
    assert_generation_context_safe(result)
    return result


def install_feedback(runtime, feedback, *, mode):
    """Opt-in before LLM generation; legacy frozen runs never acquire this mode.

    A dedicated runtime per arm is required. Do not combine this intervention
    with historical Gap stream masks, hard abstention or existing Gap prompts.
    No provider call, quota change or KnowledgeRuntime access occurs here.
    """
    if not isinstance(feedback, BankSearchFeedback):
        raise ValueError("a constructed BankSearchFeedback is required")
    feedback.direction(mode)
    if (getattr(runtime.config, "feedback_generation_policy", "legacy") != "legacy"
            or (getattr(runtime, "orchestration_context", {}) or {}).get("posterior_gap_brief")
            or any(getattr(runtime.config, name, False) for name in
                   ("structure_library_read", "structure_library_write",
                    "task_local_memory_read", "task_local_memory_write"))
            or getattr(runtime, "_bank_search_feedback", None) is not None):
        raise ValueError("Bank feedback requires a fresh dedicated unmasked runtime")
    runtime._bank_search_feedback = (feedback, mode)
