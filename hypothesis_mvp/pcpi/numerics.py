"""Central numerical compatibility helpers used by the PCPI core."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid as scipy_trapezoid


def trapezoidal_integral(values: np.ndarray) -> float:
    """Integrate a one-dimensional curve using the current NumPy API if present."""

    function = getattr(np, "trapezoid", None)
    if function is not None:
        return float(function(values))
    return float(scipy_trapezoid(values))


__all__ = ["trapezoidal_integral"]
