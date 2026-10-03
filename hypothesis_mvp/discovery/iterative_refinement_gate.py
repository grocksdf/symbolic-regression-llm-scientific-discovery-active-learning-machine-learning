"""Response-free provenance gate for a claimed posterior-driven discovery loop.

This checks supplied *production* state identities. It does not construct a
posterior, perform admission, or authorize a measured experiment. A caller must
bind each identity to its actual frozen object and supply row fingerprints from
the role datasets; a narrative log alone cannot prove the scientific model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def verify_iterative_refinement(
    cycles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require Admit(t) -> Posterior(t+1) -> Gap(t+1) in actual state records.

    Every cycle records a frozen input bank/posterior, the posterior used for
    diagnosis, the gap passed to Compose, independent admission, and the
    resulting bank/posterior. Row identities must come from exact role rows,
    not just split names. At least one admitted structure must cause a later
    diagnosis; cycles with abstention alone do not demonstrate refinement.
    """
    problems: list[str] = []
    if len(cycles) < 2:
        problems.append("at_least_two_cycles_required")
    required = (
        "bank_before", "posterior_before", "gap_bank", "gap_posterior",
        "gap_identity", "proposal_gap_identity", "bank_after",
        "posterior_after", "bank_supports_after", "admitted_supports",
        "fit_rows", "selection_rows", "gap_rows", "admission_rows",
    )
    prior: Mapping[str, object] | None = None
    all_roles: dict[str, set[str]] = {}
    linked_updates = 0
    for number, cycle in enumerate(cycles):
        missing = [key for key in required if key not in cycle]
        if missing:
            problems.append(f"cycle_{number}:missing:{','.join(missing)}")
            continue
        for key in required[:8]:
            if not isinstance(cycle[key], str) or not cycle[key]:
                problems.append(f"cycle_{number}:empty_identity:{key}")
        if cycle["gap_bank"] != cycle["bank_before"] or cycle[
                "gap_posterior"] != cycle["posterior_before"]:
            problems.append(f"cycle_{number}:gap_not_from_current_posterior")
        if cycle["proposal_gap_identity"] != cycle["gap_identity"]:
            problems.append(f"cycle_{number}:proposal_used_stale_gap")
        admitted = cycle["admitted_supports"]
        supports = cycle["bank_supports_after"]
        if (not isinstance(admitted, (list, tuple))
                or not isinstance(supports, (list, tuple))
                or any(not isinstance(x, str) or not x for x in admitted)
                or any(not isinstance(x, str) or not x for x in supports)):
            problems.append(f"cycle_{number}:invalid_support_identity")
            admitted = ()
            supports = ()
        if not set(admitted).issubset(supports):
            problems.append(f"cycle_{number}:admitted_support_missing_from_bank")
        if admitted and (cycle["bank_after"] == cycle["bank_before"]
                         or cycle["posterior_after"] == cycle["posterior_before"]):
            problems.append(f"cycle_{number}:admission_did_not_update_posterior")
        for role in ("fit_rows", "selection_rows", "gap_rows", "admission_rows"):
            rows = cycle[role]
            if (not isinstance(rows, (list, tuple)) or not rows
                    or any(not isinstance(x, str) or not x for x in rows)):
                problems.append(f"cycle_{number}:invalid_rows:{role}")
                continue
            value = set(rows)
            if len(value) != len(rows):
                problems.append(f"cycle_{number}:duplicate_rows:{role}")
            key = f"{number}:{role}"
            all_roles[key] = value
        if prior is not None:
            if (cycle["bank_before"] != prior.get("bank_after")
                    or cycle["posterior_before"] != prior.get("posterior_after")):
                problems.append(f"cycle_{number}:previous_update_not_forwarded")
            elif prior.get("admitted_supports"):
                linked_updates += 1
        prior = cycle
    # Fit/selection roles may be shared by design across cycles. Every audit
    # and admission role must be disjoint from all other roles, including its
    # own counterpart in another adaptively chosen cycle.
    fixed = {key: rows for key, rows in all_roles.items()
             if key.endswith(":fit_rows") or key.endswith(":selection_rows")}
    adaptive = {key: rows for key, rows in all_roles.items()
                if key.endswith(":gap_rows") or key.endswith(":admission_rows")}
    for key, rows in fixed.items():
        if key.endswith(":fit_rows"):
            for other, other_rows in fixed.items():
                if other.endswith(":selection_rows") and rows & other_rows:
                    problems.append("fit_selection_rows_overlap")
    for key, rows in adaptive.items():
        for other, other_rows in {**fixed, **adaptive}.items():
            if other != key and rows & other_rows:
                problems.append(f"overlapping_role_rows:{min(key, other)}:{max(key, other)}")
    if not linked_updates:
        problems.append("no_admitted_posterior_update_reached_a_later_gap")
    return {"schema": "iterative-posterior-refinement-gate-v1",
            "passed": not problems, "linked_updates": linked_updates,
            "problems": sorted(set(problems)),
            "measured_action_authorized": False}
