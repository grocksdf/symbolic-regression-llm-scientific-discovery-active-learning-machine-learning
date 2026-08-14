from __future__ import annotations

import sympy as sp


def sympy_symbols(n_features: int) -> tuple[sp.Symbol, ...]:
    """Return a tuple of symbols x0..x{n-1}, including the 1D case."""
    n_features = int(n_features)
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    syms = sp.symbols(" ".join(f"x{i}" for i in range(n_features)))
    if isinstance(syms, sp.Symbol):
        return (syms,)
    return tuple(syms)
