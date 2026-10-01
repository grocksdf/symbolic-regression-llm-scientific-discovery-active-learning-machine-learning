"""Response-free common operational-class coordinates for two nested banks.

Classes are frozen once on the expanded H0 bank. The protected core bank is
projected by exact structure ID membership, with unsupported union classes
retaining zero mass. This does not certify numerical acquisition or efficacy.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from hypothesis_mvp.pcpi.acquisition import (
    ClassPartition, fixed_partition_probabilities,
)


@dataclass(frozen=True)
class FrozenCommonClassProjection:
    union_class_ids: tuple[str, ...]
    union_partition_identity: str
    core_partition: ClassPartition
    expanded_partition: ClassPartition
    core_class_positions: tuple[int, ...]
    core_structure_ids: tuple[str, ...]
    expanded_structure_ids: tuple[str, ...]
    core_bank_identity: str
    expanded_bank_identity: str
    common_action_domain_identity: str
    common_initial_data_identity: str
    measurement_budget: int

    @property
    def stable_hash(self) -> str:
        payload = (self.union_class_ids, self.union_partition_identity,
                   self.core_partition.stable_hash,
                   self.expanded_partition.stable_hash,
                   self.core_class_positions, self.core_structure_ids,
                   self.expanded_structure_ids, self.core_bank_identity,
                   self.expanded_bank_identity,
                   self.common_action_domain_identity,
                   self.common_initial_data_identity, self.measurement_budget)
        return sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()

    def common_probabilities(self, posterior, *, bank: str) -> np.ndarray:
        """Embed each bank's fixed-class posterior into the same label space."""
        if bank == "core":
            partition = self.core_partition
            expected = self.core_structure_ids
            positions = self.core_class_positions
            bank_identity = self.core_bank_identity
        elif bank == "expanded":
            partition = self.expanded_partition
            expected = self.expanded_structure_ids
            positions = tuple(range(len(self.union_class_ids)))
            bank_identity = self.expanded_bank_identity
        else:
            raise ValueError("unknown common-class bank")
        if (posterior.bank_hash != bank_identity or
                tuple(member.structure.structure_id for member in posterior.members)
                != expected):
            raise ValueError("posterior crossed frozen common-class bank")
        masses = fixed_partition_probabilities(posterior, partition)
        common = np.zeros(len(self.union_class_ids), dtype=float)
        common[np.asarray(positions, dtype=int)] = masses
        common.setflags(write=False)
        return common


def freeze_common_class_projection(core_target, expanded_target
                                   ) -> FrozenCommonClassProjection:
    """Fix expanded H0 class labels, then project the exact core structures.

    Every core structure must occur once in the expanded bank. Both arms use
    the expanded partition's frozen H0 action-domain scale and resolution.
    Unsupported labels have zero core posterior mass, not a renamed class.
    """
    if (core_target.action_domain_identity != expanded_target.action_domain_identity
            or core_target.initial_data_identity != expanded_target.initial_data_identity
            or core_target.measurement_budget != expanded_target.measurement_budget):
        raise ValueError("common-class targets crossed domain, H0 or budget")
    core = core_target.initial_posterior
    expanded = expanded_target.initial_posterior
    core_ids = tuple(member.structure.structure_id for member in core.members)
    expanded_ids = tuple(member.structure.structure_id for member in expanded.members)
    union = expanded_target.partition
    if (len(set(core_ids)) != len(core_ids)
            or len(set(expanded_ids)) != len(expanded_ids)
            or not set(core_ids) <= set(expanded_ids)
            or len(union.structure_to_class) != len(expanded_ids)
            or len(set(union.class_ids)) != len(union.class_ids)
            or any(not indices for indices in union.member_indices)):
        raise ValueError("core structures cannot project to expanded H0 classes")
    locations = {key: index for index, key in enumerate(core_ids)}
    groups, labels, positions, assignment = [], [], [], [-1] * len(core_ids)
    for union_position, members in enumerate(union.member_indices):
        indices = tuple(sorted(locations[expanded_ids[index]] for index in members
                               if expanded_ids[index] in locations))
        if not indices:
            continue
        position = len(groups)
        groups.append(indices)
        labels.append(union.class_ids[union_position])
        positions.append(union_position)
        for member in indices:
            if assignment[member] != -1:
                raise ValueError("common-class projection repeats a structure")
            assignment[member] = position
    if any(value == -1 for value in assignment):
        raise ValueError("common-class projection omits a core structure")
    probabilities = tuple(float(sum(core.members[i].probability for i in group))
                          for group in groups)
    projected = ClassPartition(tuple(labels), tuple(groups), probabilities,
                               tuple(assignment))
    return FrozenCommonClassProjection(
        union.class_ids, union.stable_hash, projected, union,
        tuple(positions), core_ids, expanded_ids, core.bank_hash,
        expanded.bank_hash, core_target.action_domain_identity,
        core_target.initial_data_identity, core_target.measurement_budget)


def common_class_map_loss(projection: FrozenCommonClassProjection, posterior,
                          *, bank: str, true_class_id: str) -> int:
    """Zero-one loss of the MAP action on one externally specified union label."""
    if true_class_id not in projection.union_class_ids:
        raise ValueError("true class is outside frozen union target")
    masses = projection.common_probabilities(posterior, bank=bank)
    predicted = projection.union_class_ids[int(np.argmax(masses))]
    return int(predicted != true_class_id)


def finite_law_common_class_action_audit(
        projection: FrozenCommonClassProjection, core_engine, expanded_engine,
        core_posterior, expanded_posterior, action_domain, action, response_nodes,
        response_probabilities, *, true_class_id: str,
        law_identity: str) -> dict:
    """Evaluate both frozen PCPI MAP rules under the same *given* finite law.

    This is a conditional exact finite sum of floating posterior updates, not
    a continuous quadrature certificate or evidence for the supplied truth.
    The law/true class must never be obtained from a sealed reporting role.
    """
    x = np.asarray(action, dtype=float)
    domain = np.ascontiguousarray(action_domain, dtype=np.float64)
    nodes = np.asarray(response_nodes, dtype=float)
    q = np.asarray(response_probabilities, dtype=float)
    domain_identity = sha256(str(domain.shape).encode()
                             + domain.tobytes()).hexdigest()
    if (not law_identity or true_class_id not in projection.union_class_ids
            or domain.ndim != 2 or not len(domain)
            or not np.all(np.isfinite(domain))
            or domain_identity != projection.common_action_domain_identity
            or x.shape != (domain.shape[1],) or not np.all(np.isfinite(x))
            or not np.any(np.all(domain == x, axis=1))
            or nodes.ndim != 1 or len(nodes) < 2 or q.shape != nodes.shape
            or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(q))
            or np.any(q < 0) or not np.isclose(q.sum(), 1., rtol=0, atol=1e-12)
            or core_engine.design_preconditioner is not None
            or expanded_engine.design_preconditioner is not None
            or core_engine.likelihood_power != 1.
            or expanded_engine.likelihood_power != 1.
            or core_posterior.bank_hash != core_engine.bank.stable_hash
            or expanded_posterior.bank_hash != expanded_engine.bank.stable_hash):
        raise ValueError("invalid common-class finite action reference")
    before = (common_class_map_loss(projection, core_posterior, bank="core",
                                    true_class_id=true_class_id),
              common_class_map_loss(projection, expanded_posterior,
                                    bank="expanded", true_class_id=true_class_id))
    after = np.zeros(2, dtype=float)
    for response, probability in zip(nodes, q, strict=True):
        if probability == 0:
            continue
        for bank, engine, posterior, index in (
                ("core", core_engine, core_posterior, 0),
                ("expanded", expanded_engine, expanded_posterior, 1)):
            updated = engine.update_one(posterior, x, float(response))
            after[index] += probability * common_class_map_loss(
                projection, updated, bank=bank, true_class_id=true_class_id)
    return {"schema": "conditional-finite-common-pcpi-class-action-v1",
            "common_class_projection_identity": projection.stable_hash,
            "law_identity": law_identity, "true_class_id": true_class_id,
            "core_loss_before": before[0], "expanded_loss_before": before[1],
            "core_expected_loss_after": float(after[0]),
            "expanded_expected_loss_after": float(after[1]),
            "expanded_loss_reduction_vs_core_after": float(after[0] - after[1]),
            "pcpi_operational_class_finite_reference_assessed": True,
            "decision_contribution_assessed": False,
            "continuous_numerical_ranking_certified": False,
            "measured_action_authorized": False}
