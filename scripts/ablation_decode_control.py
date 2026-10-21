"""Bench-side decode-horizon control for the coupled-block ablation (ruling O2).

Why this module exists
----------------------
The three-arm child historically derived *which response rows are decoded* from the
``--condition`` argument::

    opened = order[: cuts[3] if condition == "three_arm_formula_expanded_candidate"
                    else cuts[2]]          # 75% vs 65% of the training rows

``drr_searcher`` independently requires that same condition string to imply
``iterative_posterior_refinement = True``.  Those two couplings together make
"keep the condition, remove the feedback block" an *illegal* state, so the
coupled-block ablation had no runnable Arm B.

This module turns the decode horizon into an explicit frozen benchmark control -
``decoded_response_fraction`` in the arm config - instead of a side effect of the
condition string.  When the key is absent the historical condition-derived value
is used, so already registered runs decode exactly what they always decoded.

What this does NOT do
---------------------
It never relaxes a ``hypothesis_mvp`` legality constraint and never creates the
state ``three_arm_formula_expanded_candidate + iterative_posterior_refinement=False``.
Arm B runs under a different, legal non-iterative ``three_arm_*`` condition and
merely decodes the same response rows as Arm A.  The extra rows are decoded into
``samples`` but are unreachable from ``selection``: ``split_three_arm_training_samples``
returns the ``gap_admission`` block as covariates only (``response=False``).
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np

from methods.hypothesis_mvp_pcpi.drr_adapter import (
    _ordered_indices, split_three_arm_training_samples,
)

# Historical integer cut ladder of the nine-role split.  Do not reorder: the
# indices are positional and shared with drr_adapter.three_arm_role_indices.
CUT_FRACTIONS = (0.40, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95)

ITERATIVE_FORMULA_CONDITION = "three_arm_formula_expanded_candidate"
GAP_AUDIT_CONDITION = "three_arm_l_gap_v1"

# Historical condition -> decoded fraction.  Anything else historically opened
# only development + validation + gap_audit (cuts[2]).
HISTORICAL_CONDITION_FRACTION = {ITERATIVE_FORMULA_CONDITION: CUT_FRACTIONS[3]}
DEFAULT_DECODE_FRACTION = CUT_FRACTIONS[2]

DECODE_FRACTION_KEY = "decoded_response_fraction"

DATASET_GROUPS = {
    "bio_pop_growth": "lsr_synth/bio_pop_growth",
    "chem_react": "lsr_synth/chem_react",
    "lsr_transform": "lsr_transform",
    "matsci": "lsr_synth/matsci",
    "phys_osc": "lsr_synth/phys_osc",
}

SCORED_BANK_FIELDS = ("evaluated_hypothesis_bank", "final_topk")


def decode_cuts(count: int) -> tuple[int, ...]:
    return tuple(int(round(fraction * count)) for fraction in CUT_FRACTIONS)


def resolve_decode_fraction(condition: str, explicit=None) -> float:
    """Frozen decode horizon.  ``explicit`` wins; otherwise history is replayed."""
    fraction = (HISTORICAL_CONDITION_FRACTION.get(
        str(condition), DEFAULT_DECODE_FRACTION) if explicit is None
        else float(explicit))
    if fraction not in CUT_FRACTIONS:
        raise ValueError(
            f"decode fraction {fraction!r} is not a registered cut of the "
            f"nine-role split: {CUT_FRACTIONS}")
    return fraction


def opened_row_indices(order, count: int, fraction: float) -> tuple[int, ...]:
    return tuple(order[:decode_cuts(count)[CUT_FRACTIONS.index(fraction)]])


def decoded_response_roles(fraction: float) -> tuple[str, ...]:
    roles = ["discovery_development", "discovery_validation", "gap_audit"]
    if fraction >= CUT_FRACTIONS[3]:
        roles.append("gap_admission")
    return tuple(roles)


def feedback_block_connected(condition: str) -> bool:
    """True only for the condition whose feedback block is wired to the agent."""
    return str(condition) == ITERATIVE_FORMULA_CONDITION


def gap_audit_connected(condition: str) -> bool:
    return str(condition) == GAP_AUDIT_CONDITION


def _sha_ints(values) -> str:
    return sha256(np.asarray(sorted(values), dtype=np.int64).tobytes()).hexdigest()


def _sha_floats(values) -> str:
    return sha256(np.ascontiguousarray(
        np.asarray(values, dtype=float)).tobytes()).hexdigest()


def decode_plan(*, dataset, family: str, task: str, seed: int, condition: str,
                explicit=None) -> dict:
    """Decode one coordinate and fingerprint everything the arms must share.

    ``dataset`` is the opened ``<group>/<task>/train`` HDF5 dataset.  No provider
    call, no test/OOD access: this is the same read the child performs before it
    builds the task object.
    """
    count = int(dataset.shape[0])
    X = np.asarray(dataset[:, 1:], dtype=float)
    order = _ordered_indices(count, task, seed)
    fraction = resolve_decode_fraction(condition, explicit)
    opened_indices = opened_row_indices(order, count, fraction)
    sorted_indices = sorted(opened_indices)
    opened_rows = np.asarray(dataset[sorted_indices, :])
    response_lookup = dict(zip(sorted_indices, opened_rows[:, 0], strict=True))
    y = np.zeros(count, dtype=float)
    for index in opened_indices:
        y[index] = response_lookup[index]
    samples = np.column_stack((y, X))
    roles = split_three_arm_training_samples(
        samples, task_name=str(task), seed=seed)
    admission = roles.role_row_indices["gap_admission"]
    admission_fields = [name for name in getattr(roles, "__dataclass_fields__", {})
                        if "admission" in name and name.startswith("y")]
    return {
        "family": family, "task": task, "seed": int(seed),
        "condition": str(condition),
        "decode_fraction_explicit": (
            None if explicit is None else float(explicit)),
        "decode_fraction": fraction,
        "row_count": count,
        "opened_row_count": len(opened_indices),
        "opened_indices_sha256": _sha_ints(opened_indices),
        "samples_sha256": _sha_floats(samples),
        "development_X_sha256": _sha_floats(roles.X_development),
        "development_y_sha256": _sha_floats(roles.y_development),
        "validation_X_sha256": _sha_floats(roles.X_validation),
        "validation_y_sha256": _sha_floats(roles.y_validation),
        "gap_audit_X_sha256": _sha_floats(roles.X_gap_audit),
        "gap_audit_y_sha256": _sha_floats(roles.y_gap_audit),
        "action_X_sha256": _sha_floats(roles.X_actions),
        "gap_admission_X_sha256": _sha_floats(roles.X_gap_admission),
        # The decoded gap_admission responses live in ``samples`` but no role
        # object exposes them, so they cannot reach selection/prompt/scoring.
        "gap_admission_y_fields_exposed": sorted(admission_fields),
        "gap_admission_rows_decoded": bool(
            np.all(y[list(admission)] != 0.0)) if len(admission) else False,
        "feedback_block_connected": feedback_block_connected(condition),
        "gap_audit_connected": gap_audit_connected(condition),
        "decoded_response_roles": list(decoded_response_roles(fraction)),
    }


__all__ = [name for name in dir() if not name.startswith("_")]
