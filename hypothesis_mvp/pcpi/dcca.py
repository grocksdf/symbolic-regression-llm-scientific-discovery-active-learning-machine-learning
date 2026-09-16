"""Decision-Calibrated Cross-Fitted Acquisition (DCCA) primitives.

Response-prefix only: these functions accept an explicitly bounded observed
prefix and never accept held-out arrays.  They are deliberately independent of
dataset names, seeds, and outcome-selected thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

DCCA_UTILITY = "decision-calibrated-cross-fitted-regret-reduction-v1"
DCCA_CHECKPOINT_SCHEMA = "pcpi-dcca-cross-fitted-regret-checkpoint-v1"

@dataclass(frozen=True)
class DCCAFoldPlan:
    fold_count: int
    prefix_count: int
    fold_ids: tuple[int, ...]
    plan_hash: str
    def __post_init__(self):
        if self.fold_count < 2 or self.prefix_count < self.fold_count or len(self.fold_ids) != self.prefix_count:
            raise ValueError("DCCA fold plan is invalid")
        if set(self.fold_ids) != set(range(self.fold_count)):
            raise ValueError("DCCA folds must cover every fold exactly")

def make_prefix_fold_plan(prefix_count: int, fold_count: int = 2) -> DCCAFoldPlan:
    n, k = int(prefix_count), int(fold_count)
    if n != prefix_count or k != fold_count or n < k or k < 2:
        raise ValueError("DCCA prefix/fold controls are invalid")
    ids = tuple(i % k for i in range(n))
    material = f"{n}:{k}:{ids}".encode("ascii")
    return DCCAFoldPlan(k, n, ids, hashlib.sha256(material).hexdigest())

@dataclass(frozen=True)
class DCCACalibration:
    slope: float
    intercept: float
    residual_radius: float
    train_count: int
    fold_id: int
    def interval(self, predicted: float) -> tuple[float, float]:
        if not np.isfinite(predicted): raise ValueError("predicted utility must be finite")
        center = self.intercept + self.slope * float(predicted)
        return center - self.residual_radius, center + self.residual_radius

def fit_prefix_calibration(predicted: np.ndarray, realized: np.ndarray, *, fold_id: int) -> DCCACalibration:
    x, y = np.asarray(predicted, float).reshape(-1), np.asarray(realized, float).reshape(-1)
    if len(x) != len(y) or len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("DCCA calibration inputs are invalid")
    design = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (intercept + slope * x)
    radius = float(np.max(np.abs(residual)))
    if not np.isfinite(radius): raise FloatingPointError("DCCA calibration radius is not finite")
    return DCCACalibration(float(slope), float(intercept), radius, len(x), int(fold_id))

def select_by_certified_interval(lower: np.ndarray, upper: np.ndarray) -> int:
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.ndim != 1 or hi.shape != lo.shape or len(lo) == 0 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo > hi):
        raise ValueError("DCCA utility intervals are invalid")
    leader = int(np.argmax(lo))
    if any(hi[i] > lo[leader] for i in range(len(lo)) if i != leader):
        raise RuntimeError("DCCA candidate intervals do not separate; selection abstains")
    return leader
