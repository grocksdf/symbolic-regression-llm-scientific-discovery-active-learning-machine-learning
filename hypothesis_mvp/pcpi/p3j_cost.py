"""Response-free computational ledger for P3J quadrature schedules."""

from __future__ import annotations

from dataclasses import dataclass


P3J_COST_METHOD = "static-worst-case-class-node-density-evaluation-ledger-v1"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def quadrature_orders(minimum: int, maximum: int, growth: int) -> tuple[int, ...]:
    """Return the frozen finite precision schedule, including both endpoints."""

    low = _positive_integer(minimum, "minimum")
    high = _positive_integer(maximum, "maximum")
    factor = _positive_integer(growth, "growth")
    if low < 4 or low % 2 or high < low or high % 2 or factor < 2:
        raise ValueError("P3J cost schedule is invalid")
    values = [low]
    while values[-1] < high:
        following = min(high, values[-1] * factor)
        if following % 2:
            raise ValueError("P3J cost schedule produced an odd quadrature order")
        values.append(following)
    return tuple(values)


@dataclass(frozen=True)
class P3JCostLedger:
    candidate_count: int
    ambiguity_model_count: int
    maximum_class_count: int
    maximum_leaf_count: int
    orders: tuple[int, ...]
    naive_nodes_per_leaf: int
    reused_nodes_per_leaf: int
    naive_source_class_nodes: int
    reused_source_class_nodes: int
    naive_cross_class_density_evaluations: int
    reused_cross_class_density_evaluations: int
    saved_source_class_nodes: int
    savings_fraction: float
    method: str = P3J_COST_METHOD

    def __post_init__(self) -> None:
        if (
            self.method != P3J_COST_METHOD
            or not self.orders
            or self.reused_nodes_per_leaf > self.naive_nodes_per_leaf
            or self.saved_source_class_nodes
            != self.naive_source_class_nodes - self.reused_source_class_nodes
            or not 0.0 <= self.savings_fraction < 1.0
        ):
            raise ValueError("P3J computational ledger is inconsistent")


def p3j_worst_case_cost_ledger(
    *,
    candidate_count: int,
    ambiguity_model_count: int,
    maximum_class_count: int,
    maximum_leaf_count: int,
    minimum_nodes_per_leaf: int,
    maximum_nodes_per_leaf: int,
    growth_factor: int,
) -> P3JCostLedger:
    """Count source-class nodes and cross-class density evaluations exactly."""

    candidates = _positive_integer(candidate_count, "candidate_count")
    models = _positive_integer(ambiguity_model_count, "ambiguity_model_count")
    classes = _positive_integer(maximum_class_count, "maximum_class_count")
    leaves = _positive_integer(maximum_leaf_count, "maximum_leaf_count")
    orders = quadrature_orders(
        minimum_nodes_per_leaf, maximum_nodes_per_leaf, growth_factor
    )
    naive_per_leaf = sum(order + order // 2 for order in orders)
    reused_per_leaf = orders[0] // 2 + sum(orders)
    multiplier = candidates * models * classes * leaves
    naive_nodes = multiplier * naive_per_leaf
    reused_nodes = multiplier * reused_per_leaf
    naive_density = naive_nodes * classes
    reused_density = reused_nodes * classes
    saved = naive_nodes - reused_nodes
    return P3JCostLedger(
        candidate_count=candidates,
        ambiguity_model_count=models,
        maximum_class_count=classes,
        maximum_leaf_count=leaves,
        orders=orders,
        naive_nodes_per_leaf=naive_per_leaf,
        reused_nodes_per_leaf=reused_per_leaf,
        naive_source_class_nodes=naive_nodes,
        reused_source_class_nodes=reused_nodes,
        naive_cross_class_density_evaluations=naive_density,
        reused_cross_class_density_evaluations=reused_density,
        saved_source_class_nodes=saved,
        savings_fraction=saved / naive_nodes,
    )


__all__ = [
    "P3J_COST_METHOD",
    "P3JCostLedger",
    "p3j_worst_case_cost_ledger",
    "quadrature_orders",
]
