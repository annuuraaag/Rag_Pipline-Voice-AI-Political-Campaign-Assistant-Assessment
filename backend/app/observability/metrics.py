"""Live latency percentiles for /metrics.

Each (endpoint, stage) pair keeps its most recent `window` samples in a fixed-size numpy ring
buffer: recording is O(1) with no allocation, and a summary is one vectorised percentile call
per stage. Memory is bounded at window × stages × 8 bytes per endpoint.
"""
from __future__ import annotations

import threading

import numpy as np

_QUANTILES = (50, 90, 95)


class _Ring:
    __slots__ = ("buf", "size", "next")

    def __init__(self, capacity: int) -> None:
        self.buf = np.empty(capacity, dtype=np.float64)
        self.size = 0
        self.next = 0

    def push(self, value: float) -> None:
        self.buf[self.next] = value
        self.next = (self.next + 1) % self.buf.size
        if self.size < self.buf.size:
            self.size += 1

    def values(self) -> np.ndarray:
        return self.buf[: self.size]  # order is irrelevant for percentiles


class LatencyMetrics:
    def __init__(self, window: int = 500):
        self.window = max(1, window)
        self._rings: dict[str, dict[str, _Ring]] = {}
        self._requests: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, endpoint: str, timings: dict[str, float]) -> None:
        with self._lock:
            self._requests[endpoint] = self._requests.get(endpoint, 0) + 1
            stages = self._rings.setdefault(endpoint, {})
            for stage, ms in timings.items():
                ring = stages.get(stage)
                if ring is None:
                    ring = stages[stage] = _Ring(self.window)
                ring.push(float(ms))

    def summary(self) -> dict[str, dict]:
        with self._lock:
            snapshot = {ep: {st: r.values().copy() for st, r in stages.items()} for ep, stages in self._rings.items()}
            requests = dict(self._requests)
        out: dict[str, dict] = {}
        for endpoint, stages in snapshot.items():
            report = {}
            for stage, v in stages.items():
                p50, p90, p95 = np.percentile(v, _QUANTILES)
                report[stage] = {"n": int(v.size), "p50": round(float(p50), 2), "p90": round(float(p90), 2),
                                 "p95": round(float(p95), 2), "max": round(float(v.max()), 2)}
            out[endpoint] = {"requests": requests.get(endpoint, 0), "stages": report}
        return out

    def reset(self) -> None:
        with self._lock:
            self._rings.clear()
            self._requests.clear()
