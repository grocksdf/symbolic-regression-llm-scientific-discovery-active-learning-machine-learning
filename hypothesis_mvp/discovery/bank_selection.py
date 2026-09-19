"""Response-free operational-capacity selection for a finite hypothesis bank."""
from __future__ import annotations

from hashlib import sha256
import json

from .pcpi_adapter import freeze_discovery_model, freeze_discovery_target, structural_terms


def _identity(candidate):
    return sha256(json.dumps(candidate, sort_keys=True, default=str).encode()).hexdigest()


def _required_roles(candidates):
    roles = {str(row["source"]) for row in candidates
             if str(row["source"]).startswith("engine:")}
    if any(str(row.get("origin", "")) == "llm" for row in candidates):
        roles.add("origin:llm")
    return tuple(sorted(roles))


def _roles(candidate):
    output = set()
    source = str(candidate["source"])
    if source.startswith("engine:"):
        output.add(source)
    if str(candidate.get("origin", "")) == "llm":
        output.add("origin:llm")
    return output


def select_operational_capacity_bank(candidates, initial_data, action_domain, *,
                                     n_features, prior, exploration_identity,
                                     coefficient_policy, measurement_budget,
                                     maximum_candidates):
    """Greedily maximize frozen class entropy under provenance constraints.

    Only registered H0 and acquisition covariates enter the score. Candidate
    responses, validation reporting responses and held-out objects are absent.
    """
    if (type(maximum_candidates) is not int or maximum_candidates < 3
            or maximum_candidates != 2 * measurement_budget):
        raise ValueError("bank capacity must equal twice the measurement budget")
    unique = {}
    for row in candidates:
        support = tuple(structural_terms(str(row["expression"]), n_features))
        unique.setdefault(support, dict(row))
    pool = tuple(sorted(unique.values(), key=_identity))
    required = _required_roles(pool)
    if len(pool) < 2 or maximum_candidates < len(required):
        raise ValueError("bank capacity cannot preserve required provenance roles")

    cache = {}
    def score(rows):
        key = tuple(sorted(_identity(row) for row in rows))
        if key not in cache:
            model = freeze_discovery_model(rows, n_features=n_features, prior=prior,
                exploration_identity=exploration_identity,
                coefficient_policy=coefficient_policy)
            target = freeze_discovery_target(model, initial_data, action_domain,
                measurement_budget=measurement_budget,
                expected_model_identity=model.stable_hash)
            cache[key] = (float(target.partition.entropy),
                          len(target.partition.class_ids), model.stable_hash,
                          target.stable_hash)
        return cache[key]

    selected, covered = [], set()
    remaining = list(pool)
    # First preserve every engine and LLM contribution role. For the first role
    # a second candidate is included temporarily so entropy remains defined.
    for role in required:
        options = [row for row in remaining if role in _roles(row)]
        if not options:
            raise ValueError("required hypothesis provenance role disappeared")
        ranked = []
        for row in options:
            trial = selected + [row]
            partner = None
            if len(trial) < 2:
                partner = next((item for item in remaining if item is not row), None)
                if partner is None:
                    raise ValueError("bank selection has no distinct partner")
                trial.append(partner)
            ranked.append((score(trial)[:2], _identity(row), row))
        chosen = min(ranked, key=lambda item: (-item[0][0], -item[0][1], item[1]))[2]
        selected.append(chosen); remaining.remove(chosen); covered.update(_roles(chosen))

    while remaining and len(selected) < maximum_candidates:
        ranked = [(score(selected + [row])[:2], _identity(row), row) for row in remaining]
        chosen = min(ranked, key=lambda item: (-item[0][0], -item[0][1], item[1]))[2]
        selected.append(chosen); remaining.remove(chosen)
    if not set(required).issubset(covered | set().union(*(_roles(row) for row in selected))):
        raise ValueError("selected bank lost a required provenance role")
    final_score = score(selected)
    return tuple(selected), {
        "schema": "scientific-operational-capacity-bank-v1",
        "selection_method": "greedy-frozen-class-entropy-with-provenance-coverage-v1",
        "input_support_count": len(pool), "selected_support_count": len(selected),
        "maximum_candidates": maximum_candidates, "required_roles": list(required),
        "class_entropy_nats": final_score[0], "operational_class_count": final_score[1],
        "model": final_score[2], "target": final_score[3],
        "candidate_response_accessed": False, "heldout_opened": False,
    }


__all__ = ["select_operational_capacity_bank"]
