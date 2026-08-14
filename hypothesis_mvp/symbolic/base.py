from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class SymbolicRegressor(ABC):
    """Common API for hypothesis generators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def best_expression(self) -> str:
        ...

    def info(self) -> Dict[str, Any]:
        return {}
