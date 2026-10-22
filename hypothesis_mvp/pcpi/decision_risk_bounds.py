"""Outward bounds for a supplied finite binary Student-t mixture.

Each real-line cell bounds integral(max_c joint_density_c). Student-t
unimodality bounds every density on a finite cell. A certified dominant
class integrates exactly by CDF differences; undecided cells retain lower
max(class mass) and upper total mass. Tails retain the same mass bounds.
This certifies the mathematical mixture specified by the supplied binary
numbers, not uncertainty from constructing posterior parameters in floats.
"""
from __future__ import annotations

from heapq import heappop, heappush
from importlib.metadata import version
import math
import threading

import numpy as np
from scipy.stats import t


_LOCK = threading.RLock()


def _exact(value, arb):
    n, d = float(value).as_integer_ratio()
    return arb(n) / arb(d)


def _out(value, direction):
    return float(np.nextafter(float(value), direction))


class _Mixture:
    def __init__(self, components, action, arb):
        self.arb, self.cache = arb, {}
        self.labels = components.partition.structure_to_class
        self.class_count = len(components.partition.class_ids)
        raw = [_exact(w, arb) for w in components.structure_probabilities]
        total = sum(raw, arb(0))
        self.weights = [w / total for w in raw]
        self.location_f = components.locations[:, action]
        self.scale_f = components.scales[:, action]
        self.df_f = components.degrees_freedom
        self.locations = [_exact(x, arb) for x in self.location_f]
        self.scales = [_exact(x, arb) for x in self.scale_f]
        self.df = [_exact(x, arb) for x in self.df_f]
        self.norm = [((df + 1) / 2).gamma() / (
            (df / 2).gamma() * (df * arb.pi()).sqrt() * scale)
            for df, scale in zip(self.df, self.scales)]

    def _cdf(self, point, index):
        a = self.arb
        if point == -math.inf:
            return a(0)
        if point == math.inf:
            return a(1)
        if point == self.location_f[index]:
            return a(1) / 2
        z = (_exact(point, a) - self.locations[index]) / self.scales[index]
        df, squared = self.df[index], z * z
        if squared.upper() <= df.lower():
            beta = (squared / (df + squared)).beta_lower(
                a(1) / 2, df / 2, regularized=True)
            return a(1) / 2 + (beta / 2 if point > self.location_f[index] else -beta / 2)
        beta = (df / (df + squared)).beta_lower(df / 2, a(1) / 2, regularized=True)
        return a(1) - beta / 2 if point > self.location_f[index] else beta / 2

    def endpoint(self, point):
        if point not in self.cache:
            cdf = [self._cdf(point, i) for i in range(len(self.weights))]
            pdf = []
            for i, (df, scale, location) in enumerate(zip(self.df, self.scales, self.locations)):
                if not math.isfinite(point):
                    pdf.append(self.arb(0))
                else:
                    z = (_exact(point, self.arb) - location) / scale
                    pdf.append(self.norm[i] * (1 + z * z / df) ** (-(df + 1) / 2))
            if not all(v.is_finite() for v in (*cdf, *pdf)):
                raise ArithmeticError("Student-t ball evaluation unresolved")
            self.cache[point] = cdf, pdf
        return self.cache[point]

    def class_sums(self, values):
        out = [self.arb(0) for _ in range(self.class_count)]
        for i, value in enumerate(values):
            out[self.labels[i]] += self.weights[i] * value
        return out

    def cell(self, left, right):
        lc, lp = self.endpoint(left)
        rc, rp = self.endpoint(right)
        masses = self.class_sums([b - a for a, b in zip(lc, rc)])
        lo = max(0., max(_out(m.lower(), -math.inf) for m in masses))
        hi = min(1., _out(sum(masses, self.arb(0)).upper(), math.inf))
        if math.isfinite(left) and math.isfinite(right):
            lower = self.class_sums([
                a.lower() if a.lower() < b.lower() else b.lower()
                for a, b in zip(lp, rp)])
            upper = self.class_sums([
                self.norm[i].upper() if left <= loc <= right else
                (lp[i].upper() if lp[i].upper() > rp[i].upper() else rp[i].upper())
                for i, loc in enumerate(self.location_f)])
            leader = max(range(self.class_count), key=lambda c: float(lower[c].lower()))
            if all(lower[leader].lower() >= upper[c].upper()
                   for c in range(self.class_count) if c != leader):
                lo = max(lo, _out(masses[leader].lower(), -math.inf))
                hi = min(hi, _out(masses[leader].upper(), math.inf))
            else:
                width = _exact(right, self.arb) - _exact(left, self.arb)
                upper_density = max(v.upper() for v in upper)
                hi = min(hi, _out((width * upper_density).upper(), math.inf))
        if not 0 <= lo <= hi <= 1:
            raise ArithmeticError("invalid class accuracy enclosure")
        return lo, hi


def _action_bound(components, action, arb, epsabs, epsrel, maximum_cells):
    model = _Mixture(components, action, arb)
    points = []
    tail = min(epsabs / 64., 1e-8)
    for loc, scale, df in zip(model.location_f, model.scale_f, model.df_f):
        quantiles = t.ppf([tail, .01, .1, .5, .9, .99, 1 - tail], df)
        points.extend(float(loc + scale * q) for q in quantiles if np.isfinite(loc + scale * q))
    points = sorted(set(points))
    if not points:
        raise ArithmeticError("finite mixture integration domain unavailable")
    heap, serial = [], 0
    for left, right in zip([-math.inf, *points], [*points, math.inf]):
        lo, hi = model.cell(left, right)
        heappush(heap, (-(hi - lo), serial, left, right, lo, hi))
        serial += 1
    while len(heap) < maximum_cells:
        lower = math.fsum(row[4] for row in heap)
        upper = math.fsum(row[5] for row in heap)
        if upper - lower <= epsabs + epsrel * abs(lower):
            break
        _, _, left, right, lo, hi = heappop(heap)
        if not math.isfinite(left) or not math.isfinite(right):
            middle = (right - max(1., abs(right)) if not math.isfinite(left)
                      else left + max(1., abs(left)))
        else:
            middle = left / 2 + right / 2
        if not left < middle < right or not math.isfinite(middle):
            heappush(heap, (0., serial, left, right, lo, hi))
            break
        for a, b in ((left, middle), (middle, right)):
            lo, hi = model.cell(a, b)
            heappush(heap, (-(hi - lo), serial, a, b, lo, hi))
            serial += 1
    lower = _out(math.fsum(row[4] for row in heap), -math.inf)
    upper = _out(math.fsum(row[5] for row in heap), math.inf)
    priors = model.class_sums([arb(1) for _ in model.weights])
    prior_lo = max(_out(v.lower(), -math.inf) for v in priors)
    prior_hi = max(_out(v.upper(), math.inf) for v in priors)
    return max(0., _out(lower - prior_hi, -math.inf)), min(
        _out(1. - prior_lo, math.inf), _out(upper - prior_lo, math.inf))


def decision_risk_intervals(components, *, epsabs, epsrel, maximum_cells=1024):
    if version("python-flint") != "0.8.0":
        raise RuntimeError("decision-risk enclosure requires python-flint==0.8.0")
    from flint import arb, ctx
    with _LOCK, ctx.workprec(256):
        bounds = [_action_bound(components, i, arb, epsabs, epsrel, maximum_cells)
                  for i in range(components.locations.shape[1])]
    return np.asarray(bounds, dtype=float).T
