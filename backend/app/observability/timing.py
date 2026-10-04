"""Wall-clock timing of pipeline stages within one request.

    timer = StageTimer()
    with timer.stage("embed"):
        ...
    timer.mark("first_token")      # elapsed since the timer was created
    timer.as_dict()                # {"embed": 7.1, "first_token": 412.0, "total": 455.3}

Stages are durations (repeated stages add up); marks are points in time since the start
(first write wins, so a mark placed inside a token loop records the first token only).
Everything is measured with `perf_counter_ns`; nothing is estimated.
"""
from __future__ import annotations

from time import perf_counter_ns

_NS_PER_MS = 1_000_000


class _Stage:
    __slots__ = ("_timer", "_name", "_start")

    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._start = 0

    def __enter__(self) -> None:
        self._start = perf_counter_ns()

    def __exit__(self, *_exc) -> bool:
        self._timer._add(self._name, perf_counter_ns() - self._start)
        return False  # never swallow exceptions


class StageTimer:
    __slots__ = ("_origin", "_durations", "_points")

    def __init__(self) -> None:
        self._origin = perf_counter_ns()
        self._durations: dict[str, int] = {}   # stage → nanoseconds spent
        self._points: dict[str, int] = {}      # mark → nanoseconds after origin

    def stage(self, name: str) -> _Stage:
        """Context manager timing one stage; the duration is added even if the block raises."""
        return _Stage(self, name)

    def _add(self, name: str, ns: int) -> None:
        self._durations[name] = self._durations.get(name, 0) + ns

    def record(self, name: str, ms: float) -> None:
        """Add a duration measured elsewhere (e.g. by another process)."""
        self._add(name, int(ms * _NS_PER_MS))

    def mark(self, name: str) -> None:
        if name not in self._points:
            self._points[name] = perf_counter_ns() - self._origin

    def elapsed_ms(self) -> float:
        return (perf_counter_ns() - self._origin) / _NS_PER_MS

    def as_dict(self) -> dict[str, float]:
        out = {name: round(ns / _NS_PER_MS, 2) for name, ns in self._durations.items()}
        out.update((name, round(ns / _NS_PER_MS, 2)) for name, ns in self._points.items())
        out["total"] = round(self.elapsed_ms(), 2)
        return out
