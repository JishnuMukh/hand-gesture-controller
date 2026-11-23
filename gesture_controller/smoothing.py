"""Volume smoothing.

Two stages:
1. ``OneEuroFilter`` on the raw pinch-derived target removes landmark jitter.
   It adapts its cutoff to speed: heavy smoothing when the hand is still,
   light smoothing (low lag) when it moves quickly.
2. ``VolumeSmoother`` slew-limits the filtered target so the system volume
   never changes faster than ``max_rate_per_s``, and keeps a short history so
   the level from just before a release can be restored.
"""

from __future__ import annotations

import math
from collections import deque


def _alpha(cutoff_hz: float, dt: float) -> float:
    if dt <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Casiez et al., "1€ Filter", CHI 2012."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0) -> None:
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_cutoff = d_cutoff_hz
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = float(x), t
            return self._x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        dx = (x - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += _alpha(cutoff, dt) * (x - self._x)
        return self._x


class VolumeSmoother:
    def __init__(self, max_rate_per_s: float, history_s: float = 1.0) -> None:
        self.max_rate = max_rate_per_s
        self.history_s = history_s
        self.value: float | None = None
        self._last_t: float | None = None
        self._history: deque[tuple[float, float]] = deque()

    def reset(self, value: float, now: float) -> None:
        self.value = float(value)
        self._last_t = now
        self._history.clear()
        self._history.append((now, self.value))

    def update(self, target: float, now: float) -> float:
        if self.value is None or self._last_t is None:
            self.reset(target, now)
            return self.value  # type: ignore[return-value]
        dt = max(now - self._last_t, 0.0)
        self._last_t = now
        max_step = self.max_rate * dt
        step = max(-max_step, min(max_step, target - self.value))
        self.value = min(100.0, max(0.0, self.value + step))
        self._history.append((now, self.value))
        while self._history and now - self._history[0][0] > self.history_s:
            self._history.popleft()
        return self.value

    def value_at(self, t: float) -> float | None:
        """Most recent value recorded at or before time ``t``."""
        best = None
        for ts, v in self._history:
            if ts <= t:
                best = v
            else:
                break
        if best is None and self._history:
            best = self._history[0][1]
        return best


def map_ratio_to_volume(ratio: float, lo: float, hi: float) -> float:
    if hi <= lo:
        raise ValueError("pinch_max_ratio must be greater than pinch_min_ratio")
    return 100.0 * min(1.0, max(0.0, (ratio - lo) / (hi - lo)))
