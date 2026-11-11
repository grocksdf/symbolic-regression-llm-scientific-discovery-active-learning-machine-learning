"""Frozen complete proposals shared by program and LLM selection policies.

Only fitting observations enter features. The fixed eight-dimensional feature
map describes a proposed change's behaviour, not an AST-label lookup table or
a learned causal attribution. Independent evaluation is deliberately absent.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass

import numpy as np

from hypothesis_mvp.data.roles import DataRole
from hypothesis_mvp.pcpi.reference.basis import design_matrix
from hypothesis_mvp.pcpi.reference.expanded_formula_basis import compile_fixed_formula_support, _run
from .counterfactual_bank_symbolic_invention import _ast_paths, _node_at_path
from .region_site_attribution import digest


FEATURE_NAMES = (
    "global_change_alignment", "local_change_alignment", "local_change_energy",
    "bank_behaviour_novelty", "relative_ast_size", "edited_subtree_fraction",
    "defect_focus_interaction", "behaviour_available",
)
SCHEMA = "frozen-structural-plan-pool-v1"


@dataclass(frozen=True)
class StructuralPlan:
    expression: str
    parent_expression: str
    region_id: str
    operation: str
    source: str
    lineage_id: str
    site_path: tuple[int, ...] | None = None
    prompt_hash: str = ""
    response_hash: str = ""

    def __post_init__(self):
        if (not all(isinstance(v, str) and v.strip() for v in (
                self.expression, self.parent_expression, self.region_id,
                self.operation, self.source, self.lineage_id))
                or self.source not in {"program", "llm"}
                or (self.site_path is not None and (type(self.site_path) is not tuple
                    or any(type(i) is not int or i < 0 for i in self.site_path)))
                or (self.source == "llm" and not (self.prompt_hash and self.response_hash))):
            raise ValueError("complete immutable structural plan and provenance required")

    @property
    def identity(self):
        return digest(asdict(self))


def plans_from_proposal_batch(batch, *, parent_expressions, region_id):
    """Adapt genuine ProposalRuntime outputs; never use mock_llm_expression.

    Historical cost rows without transport/response identity are insufficient.
    A failed provider request cannot turn into an empty valid proposal pool.
    Partial protocol rejections remain in the returned audit.
    """
    requests = batch.telemetry.get("provider_requests", ())
    if batch.reason == "provider_or_protocol_failure" or not requests:
        raise ValueError("provider/protocol failure or missing transport evidence")
    attempts = [a for r in requests for a in r.get("provider_outcomes", ())]
    if (not attempts or any(not r.get("provider_all_attempts_preserved") for r in requests)
            or any(type(a.get("provider_http_status")) is not int
                   or not 200 <= a["provider_http_status"] < 300
                   or a.get("provider_error") for a in attempts)):
        raise ValueError("proposal pool requires preserved successful transport attempts")
    plans = []
    for candidate in batch.candidates:
        if (candidate.parent_hash not in parent_expressions
                or candidate.prompt_hash != batch.prompt_hash
                or candidate.response_hash != batch.response_hash):
            raise ValueError("LLM candidate parent/response binding mismatch")
        plans.append(StructuralPlan(candidate.equation, parent_expressions[candidate.parent_hash],
            region_id, candidate.action, "llm", candidate.lineage_id,
            prompt_hash=candidate.prompt_hash, response_hash=candidate.response_hash))
    audit = {"batch_prompt_hash": batch.prompt_hash, "batch_response_hash": batch.response_hash,
             "logical_calls": len(requests), "provider_attempts": len(attempts),
             "protocol_valid": batch.protocol_valid, "reason": batch.reason,
             "rejections": batch.telemetry.get("candidate_validation_rejections", []),
             "plan_ids": [p.identity for p in plans]}
    # Whitelist cost fields. Do not serialize arbitrary provider metadata/secrets.
    for field in ("prompt_tokens", "completion_tokens", "elapsed_s"):
        values = [a.get("provider_" + field) for a in attempts]
        audit[field] = sum(values) if all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0
                                        for v in values) else None
    return tuple(plans), audit


def plan_from_program_request(editor, request, *, region_id):
    if request.mock_llm_expression is not None or request.operator == "llm_proposed_edit":
        raise ValueError("program proposals cannot impersonate LLM responses")
    return StructuralPlan(editor._raw_edit(request), request.parent_expression, region_id,
                          request.operator, "program", request.identity, request.site_path)


def _unit(values, *, centered=False):
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    scaled = values / scale if scale else np.zeros_like(values)
    if centered:
        scaled = scaled - np.mean(scaled)
    norm = float(np.linalg.norm(scaled))
    return scaled / norm if norm else np.zeros_like(scaled)


def _cosine(a, b, *, centered=False):
    # Clip only dot-product roundoff for bounded FEATURES, never utilities.
    return float(np.clip(_unit(a, centered=centered) @ _unit(b, centered=centered), -1., 1.))


def _raw_values(expression, inputs):
    result = np.broadcast_to(_run(ast.parse(expression, mode="eval").body, inputs), (len(inputs),))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite raw formula behaviour")
    return result


def plan_features(plan, *, fit_inputs, direction, bank_locations, mask):
    """Pre-fit features; no candidate scores, admissions, or LLM self-ratings.

    Grammar and sites must already be validated by freeze_plan_pool. Domain
    failure is an explicit unavailable-behaviour flag, not an efficacy score.
    """
    parent = ast.parse(plan.parent_expression, mode="eval").body
    candidate = ast.parse(plan.expression, mode="eval").body
    psize, csize = len(list(ast.walk(parent))), len(list(ast.walk(candidate)))
    site_fraction = (len(list(ast.walk(_node_at_path(parent, plan.site_path)))) / psize
                     if plan.site_path is not None else 1.)
    size_change = (csize - psize) / (csize + psize)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            p, c = _raw_values(plan.parent_expression, fit_inputs), _raw_values(plan.expression, fit_inputs)
            scale = max(float(np.max(np.abs(p))), float(np.max(np.abs(c))))
            change = c / scale - p / scale if scale else np.zeros_like(c)
            unit_change = _unit(change)
            focus = float(unit_change[mask] @ unit_change[mask])
            unit_direction = _unit(direction)
            defect_fraction = float(unit_direction[mask] @ unit_direction[mask])
            novelty = 1. - max(abs(_cosine(c, v, centered=True)) for v in bank_locations.T)
            features = (_cosine(direction, change), _cosine(direction[mask], change[mask]), focus,
                        novelty, size_change, site_fraction,
                        (defect_fraction - float(np.mean(mask))) * focus, 1.)
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        features = (0., 0., 0., 0., size_change, site_fraction, 0., 0.)
    if not np.all(np.isfinite(features)):
        raise ValueError("nonfinite plan feature contract")
    return tuple(float(v) for v in features)


def freeze_plan_pool(plans, *, fit, fit_row_ids, forbidden_row_ids, feedback,
                     bank_locations, core_rows, regions, task_identity):
    """Compile all proposals once, preserve every rejection and support alias."""
    fit_ids, forbidden = tuple(fit_row_ids), tuple(forbidden_row_ids)
    loc = np.asarray(bank_locations, dtype=float)
    if (fit.role is not DataRole.DEVELOPMENT or not task_identity or not core_rows
            or fit.fingerprint != feedback.search_data_identity
            or fit_ids != tuple(feedback.search_row_ids) or len(set(fit_ids)) != len(fit_ids)
            or len(fit_ids) != len(fit.X) or not forbidden or set(fit_ids).intersection(forbidden)
            or loc.ndim != 2 or loc.shape[0] != len(fit.X) or not loc.shape[1]
            or not np.all(np.isfinite(loc))):
        raise ValueError("plan features require isolated bound fitting data")
    n_features = fit.X.shape[1]
    core_supports = {digest(compile_fixed_formula_support(r["expression"], n_features)) for r in core_rows}
    masks = {}
    for region in regions:
        if (region["id"] in masks or type(region["axis"]) is not int
                or not 0 <= region["axis"] < n_features or region["side"] not in {"le", "gt"}
                or not np.isfinite(region["cut"])):
            raise ValueError("invalid frozen feature region")
        mask = fit.X[:, region["axis"]] <= region["cut"]
        if region["side"] == "gt": mask = ~mask
        if not np.any(mask): raise ValueError("empty feature region")
        masks[region["id"]] = mask
    if not masks: raise ValueError("frozen regions required")
    plans = tuple(plans)
    if not plans or len({p.identity for p in plans}) != len(plans):
        raise ValueError("nonempty unique proposal identities required")
    context_id = digest({"fit": fit.fingerprint, "fit_ids": fit_ids, "feedback": feedback.identity,
                         "bank": feedback.bank_identity, "regions": regions, "features": FEATURE_NAMES,
                         "bank_locations": loc.tolist(), "task": task_identity, "core": core_rows})
    rows, rejected, supports = [], [], {}
    for plan in sorted(plans, key=lambda p: (p.expression, p.parent_expression, p.identity)):
        if plan.region_id not in masks: raise ValueError("proposal references an unknown region")
        try:
            support = compile_fixed_formula_support(plan.expression, n_features)
            compile_fixed_formula_support(plan.parent_expression, n_features)
            if (plan.site_path is not None
                    and plan.site_path not in _ast_paths(ast.parse(plan.parent_expression, mode="eval").body)):
                raise ValueError("proposal references an invalid edit site")
        except (ValueError, SyntaxError) as error:
            rejected.append({"plan": asdict(plan), "plan_id": plan.identity,
                             "stage": "compile", "reason": str(error)})
            continue
        support_id = digest(support)
        if support_id in supports:
            rejected.append({"plan": asdict(plan), "plan_id": plan.identity, "stage": "alias",
                             "representative": supports[support_id], "support_id": support_id})
            continue
        supports[support_id] = plan.identity
        features = plan_features(plan, fit_inputs=fit.X, direction=np.asarray(feedback.location_score),
                                 bank_locations=loc, mask=masks[plan.region_id])
        rows.append({"plan_id": plan.identity, "plan": asdict(plan), "features": features,
                     "support_id": support_id, "core_duplicate": support_id in core_supports})
    value = {"schema": SCHEMA, "task_identity": task_identity, "bank_identity": feedback.bank_identity,
             "context_identity": context_id, "feature_names": FEATURE_NAMES, "rows": rows,
             "rejections": rejected, "input_count": len(plans), "regions": regions,
             "source_plan_ids": sorted(p.identity for p in plans), "fit_row_ids": fit_ids,
             "fit_fingerprint": fit.fingerprint, "core_identity": digest(core_rows),
             "feedback_identity": feedback.identity, "n_features": n_features}
    return {**value, "identity": digest(value)}


def candidate_domain_status(expression, *, n_features, domains):
    """Only known mathematical undefinedness is a candidate rejection.

    Invalid domains, compiler/programmer errors, linear algebra failures and
    arbitrary exceptions propagate. They must never train the policy as zero.
    """
    support = compile_fixed_formula_support(expression, n_features)
    if not domains: raise ValueError("registered covariate domains required")
    for domain in domains:
        x = np.asarray(domain, dtype=float)
        if x.ndim != 2 or not len(x) or x.shape[1] != n_features or not np.all(np.isfinite(x)):
            raise ValueError("invalid registered covariate domain")
        try:
            design_matrix(x, support)
        except ValueError as error:
            if str(error) not in {"fixed formula undefined on registered domain", "nonfinite fixed formula design column"}:
                raise
            return False, str(error)
    return True, None
