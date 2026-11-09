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
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from hypothesis_mvp.data.roles import DataRole, RoleDataset
from .proposal_calibration import assert_generation_context_safe
from .bank_diagnostic_context import (
    MULTISCALE_FEEDBACK_SCHEMA, build_diagnostic_snapshot,
    project_diagnostic_scope, validate_diagnostic_axes, validate_diagnostic_view,
)


BANK_PROPOSAL_FEEDBACK_SCHEMA = "bank-complementary-proposal-feedback-v1"


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
    bank_inventory_json: str = ""
    diagnostics_json: str = ""

    @property
    def identity(self):
        fields = dict(self.__dict__)
        if not self.bank_inventory_json:
            fields.pop("bank_inventory_json")  # Preserve the legacy v1 identity.
        if not self.diagnostics_json:
            fields.pop("diagnostics_json")
        return _hash(fields)

    def direction(self, mode):
        if mode == "bank":
            return np.asarray(self.location_score)
        if mode == "residual":
            return np.asarray(self.residual)
        if mode == "blind":
            return np.zeros(len(self.search_row_ids))
        raise ValueError("unknown search feedback mode")

    def prompt_payload(self, mode="bank", *, diagnostic_scope=None):
        if mode not in {"bank", "residual", "blind"}:
            raise ValueError("unknown search feedback mode")
        fields = {"bank": ("mean_log_density", "mean_location_score"),
                  "residual": ("mean_residual",), "blind": ()}[mode]
        regions = [{key: value for key, value in row.items()
                    if key in ("axis", "side", "cut", "count", *fields)}
                   for row in json.loads(self.regions_json)] if fields else []
        payload = {"schema": "bank-search-feedback-v1", "mode": mode,
                "signal_identity": self.identity if mode != "blind" else None,
                "source_role": "opened-generation-search-only", "regions": regions,
                "instruction": "Use this local diagnostic to propose structural changes. "
                               "Positive mean_location_score favors a small upward shift of "
                               "the Bank predictive locations; negative favors downward. "
                               "A residual is observed target minus fitted MAP prediction. "
                               "These are not gradients of arbitrary AST replacements. "
                               "Preserve registered grammar and search rights; the signal "
                               "does not select or certify the final predictor."}
        if self.bank_inventory_json:
            inventory = json.loads(self.bank_inventory_json)
            payload.update(schema=BANK_PROPOSAL_FEEDBACK_SCHEMA,
                           bank_inventory=inventory,
                           bank_inventory_identity=_hash(inventory))
            payload["instruction"] += (
                " The common bank_inventory lists the existing structural supports, "
                "not their truth or posterior ranking. Its expressions display "
                "unweighted basis sums, not fitted predictions; outer amplitudes "
                "are refit by the numerical engine. Seek a falsifiable structure "
                "whose predictive behavior could complement this whole Bank. "
                "You may use a parent edit or a complete new structure when the "
                "unchanged action grammar and new-skeleton quota permit it. "
                "Treat diagnostics as fallible search evidence, not mandatory "
                "directions or an instruction to abstain. Explain the proposed "
                "complementarity in rationale; that explanation is not evidence.")
        if diagnostic_scope is not None:
            if not self.bank_inventory_json or not self.diagnostics_json:
                raise ValueError("multiscale feedback requires a frozen inventory and diagnostic design")
            if mode != ("blind" if diagnostic_scope == "none" else "bank"):
                raise ValueError("multiscale diagnostics require Bank or blind information")
            payload.update(schema=MULTISCALE_FEEDBACK_SCHEMA,
                           **project_diagnostic_scope(json.loads(self.diagnostics_json), diagnostic_scope))
            payload["instruction"] = (
                "Use the common bank_inventory as the existing structural comparison set. "
                "Its expressions are unweighted basis sums with numerical amplitudes fitted separately. "
                "Visible summaries describe fitting observations only. mean_location_score is the "
                "derivative for a common small shift of all Bank predictive locations. Positive "
                "suggests upward, negative downward; it is a fallible proposal cue. "
                "Compare mean_abs_location_score and positive/negative fractions to detect "
                "cancellation. Opposite regional directions may require a shape change. "
                "global_local_links describe regional deviations from the whole dataset. "
                "Each axis partition covers the same observations, so counts across axes overlap. "
                "Small regional counts offer limited support; these summaries carry no confidence "
                "certificate or verified causal explanation. Judge a proposed structure by how it "
                "could complement the entire Bank, under the registered grammar and quota. "
                "Describe the diagnostic basis and a testable explanation in rationale. "
                "You may disagree with a diagnostic direction. Independent evaluation determines "
                "predictive value after generation. Scope controls visibility, not search rights.")
            validate_diagnostic_view(payload)
        return payload


def feedback_from_posterior(engine, posterior, search, *, search_row_ids,
                            forbidden_row_ids, bank_identity,
                            include_bank_inventory=False, diagnostic_axes=None):
    """Read only opened generation/search roles; require explicit row exclusion.

    forbidden_row_ids are opaque registered admission/reporting indices, not
    response hashes. The caller must bind these IDs to actual dataset rows.
    """
    if type(include_bank_inventory) is not bool:
        raise ValueError("bank inventory must be explicitly enabled or disabled")
    if diagnostic_axes is not None:
        validate_diagnostic_axes(diagnostic_axes, search.X.shape[1])
        if not include_bank_inventory or search.role is not DataRole.DEVELOPMENT:
            raise ValueError("multiscale feedback requires fitting development rows and Bank inventory")
    ids, forbidden = tuple(search_row_ids), tuple(forbidden_row_ids)
    if (search.role not in {DataRole.DEVELOPMENT, DataRole.VALIDATION}
            or len(ids) != len(search.X) or len(set(ids)) != len(ids)
            or not forbidden or any(not isinstance(v, str) or not v for v in (*ids, *forbidden))
            or set(ids).intersection(forbidden) or not isinstance(bank_identity, str)
            or not bank_identity or posterior.bank_hash != engine.bank.stable_hash):
        raise ValueError("feedback needs a bound Bank and isolated search rows")
    locations, scales, degrees = [], [], []
    for member in posterior.members:
        location, squared_scale, df = engine.predictive_location_scale(member, search.X)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            locations.append(location)
            scales.append(np.sqrt(squared_scale))
            degrees.append(df)
    lpd, score, residual = mixture_location_feedback(
        np.column_stack(locations), np.column_stack(scales), degrees,
        posterior.log_probabilities, search.y)
    regions, diagnostics = [], ""
    if diagnostic_axes is None:
        # Keep historical summaries and identities exactly unchanged.
        for axis in range(search.X.shape[1]):
            cut = float(np.median(search.X[:, axis]))
            for side, mask in (("le", search.X[:, axis] <= cut), ("gt", search.X[:, axis] > cut)):
                if not np.any(mask):
                    continue
                regions.append({"axis": axis, "side": side, "cut": cut, "count": int(np.sum(mask)),
                                "mean_log_density": float(np.mean(lpd[mask])),
                                "mean_location_score": float(np.mean(score[mask])),
                                "mean_residual": float(np.mean(residual[mask]))})
    else:
        snapshot = build_diagnostic_snapshot(search.X, lpd, score, axes=diagnostic_axes)
        diagnostics = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        for region in snapshot["regions"]:
            mask = search.X[:, region["axis"]] <= region["cut"]
            if region["side"] == "gt":
                mask = ~mask
            regions.append({**region, "mean_residual": math.fsum(
                float(value) / region["count"] for value in residual[mask])})
    inventory = ""
    if include_bank_inventory:
        from hypothesis_mvp.pcpi.reference.expanded_formula_basis import decode_fixed_formula_term
        def display(term):
            if term.startswith("formula_ast_v1:"):
                return "(" + decode_fixed_formula_term(term, search.X.shape[1]) + ")"
            return "1" if term == "intercept" else term
        # All registered structures, in response-independent order. No posterior
        # ranking, parameters, raw rows or independent outcome enter this packet.
        structures = [{"structure_id": member.structure.structure_id,
                       "expression": " + ".join(display(term) for term in member.structure.basis_terms),
                       "basis_terms": list(member.structure.basis_terms)}
                      for member in posterior.members]
        inventory = json.dumps(sorted(structures, key=lambda row: row["structure_id"]),
                               sort_keys=True, allow_nan=False)
    return BankSearchFeedback(bank_identity, ids, search.fingerprint,
                              tuple(map(float, lpd)), tuple(map(float, score)),
                              tuple(map(float, residual)), json.dumps(regions, allow_nan=False), inventory, diagnostics)


def validate_bank_prompt_feedback(payload):
    """Reject malformed opt-in diagnostics before transport; do not repair them."""
    assert_generation_context_safe(payload)
    if not isinstance(payload, dict):
        raise ValueError("Bank feedback must be a typed object")
    common = {"schema", "mode", "signal_identity", "source_role", "regions", "instruction"}
    multiscale = payload.get("schema") == MULTISCALE_FEEDBACK_SCHEMA
    rich = payload.get("schema") in {BANK_PROPOSAL_FEEDBACK_SCHEMA, MULTISCALE_FEEDBACK_SCHEMA}
    expected = common | ({"bank_inventory", "bank_inventory_identity"} if rich else set())
    if multiscale:
        expected |= {"diagnostic_scope", "global_summary", "global_local_links"}
    mode = payload.get("mode")
    if (set(payload) != expected or payload.get("schema") not in
            {"bank-search-feedback-v1", BANK_PROPOSAL_FEEDBACK_SCHEMA, MULTISCALE_FEEDBACK_SCHEMA}
            or mode not in {"bank", "residual", "blind"}
            or payload.get("source_role") != "opened-generation-search-only"
            or not isinstance(payload.get("instruction"), str)):
        raise ValueError("invalid Bank feedback schema or information condition")
    signal = payload["signal_identity"]
    if ((mode == "blind" and signal is not None)
            or (mode != "blind" and (not isinstance(signal, str) or not signal))):
        raise ValueError("invalid Bank feedback signal identity")
    regions = payload["regions"]
    if not isinstance(regions, list) or (mode == "blind" and regions):
        raise ValueError("blind condition cannot contain feedback regions")
    metrics = {"bank": {"mean_log_density", "mean_location_score"},
               "residual": {"mean_residual"}, "blind": set()}[mode]
    if multiscale:
        validate_diagnostic_view(payload)
    for region in (() if multiscale else regions):
        if (not isinstance(region, dict) or set(region) != {"axis", "side", "cut", "count"} | metrics
                or type(region.get("axis")) is not int or region["axis"] < 0
                or region.get("side") not in {"le", "gt"}
                or type(region.get("count")) is not int or region["count"] <= 0
                or any(type(region.get(key)) not in {int, float}
                       or not np.isfinite(region[key]) for key in {"cut"} | metrics)):
            raise ValueError("invalid or cross-condition Bank region diagnostic")
    if rich:
        inventory = payload["bank_inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("complete Bank structure inventory required")
        for row in inventory:
            if (not isinstance(row, dict) or set(row) != {"structure_id", "expression", "basis_terms"}
                    or any(not isinstance(row[key], str) or not row[key]
                           for key in ("structure_id", "expression"))
                    or not isinstance(row["basis_terms"], list) or not row["basis_terms"]
                    or any(not isinstance(term, str) or not term for term in row["basis_terms"])):
                raise ValueError("invalid Bank structure inventory")
        if (len({row["structure_id"] for row in inventory}) != len(inventory)
                or payload["bank_inventory_identity"] != _hash(inventory)):
            raise ValueError("Bank inventory identity mismatch")
    return rich


def validate_feedback_source(feedback, X, y):
    """Bind an installed diagnostic to the actual generation-fitting arrays."""
    data = RoleDataset(DataRole.DEVELOPMENT, X, y)
    if (data.fingerprint != feedback.search_data_identity
            or len(data.X) != len(feedback.search_row_ids)):
        raise ValueError("Bank feedback does not match generation fitting rows")


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


def attach_feedback(context, feedback, *, mode, diagnostic_scope=None):
    """Explicit LLM integration hook; no grammar/quota modification or transport."""
    assert_generation_context_safe(context)
    if (context.get("posterior_gap_brief") is not None
            or (context.get("scientist_orchestration") or {}).get("posterior_gap_brief") is not None):
        raise ValueError("historical Gap prompts cannot be mixed with Bank-location guidance")
    result = json.loads(json.dumps(context, allow_nan=False))
    result["bank_predictive_search_feedback"] = feedback.prompt_payload(mode, diagnostic_scope=diagnostic_scope)
    validate_bank_prompt_feedback(result["bank_predictive_search_feedback"])
    assert_generation_context_safe(result)
    return result


def install_feedback(runtime, feedback, *, mode, diagnostic_scope=None):
    """Opt-in before LLM generation; legacy frozen runs never acquire this mode.

    A dedicated runtime per arm is required. Do not combine this intervention
    with historical Gap stream masks, hard abstention or existing Gap prompts.
    No provider call, quota change or KnowledgeRuntime access occurs here.
    """
    if not isinstance(feedback, BankSearchFeedback):
        raise ValueError("a constructed BankSearchFeedback is required")
    feedback.direction(mode)
    if diagnostic_scope is not None:
        validate_bank_prompt_feedback(feedback.prompt_payload(mode, diagnostic_scope=diagnostic_scope))
    if (getattr(runtime.config, "feedback_generation_policy", "legacy") != "legacy"
            or (getattr(runtime, "orchestration_context", {}) or {}).get("posterior_gap_brief")
            or any(getattr(runtime.config, name, False) for name in
                   ("structure_library_read", "structure_library_write",
                    "task_local_memory_read", "task_local_memory_write"))
            or getattr(runtime, "_bank_search_feedback", None) is not None):
        raise ValueError("Bank feedback requires a fresh dedicated unmasked runtime")
    runtime._bank_search_feedback = (feedback, mode)
    if diagnostic_scope is not None:
        runtime._bank_diagnostic_scope = diagnostic_scope
