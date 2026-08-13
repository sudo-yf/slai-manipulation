"""Deterministic WujiHand command loop independent of teleoperation source."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from .client import WujiHandClient
from .filters import OneEuroFilter
from .safety import SafeCommandLimiter


def advance_periodic_deadline(deadline: float, period: float, now: float) -> tuple[float, int]:
    if period <= 0:
        raise ValueError("period must be positive")
    missed = max(0, int((now - deadline) // period) + 1) if now >= deadline else 0
    return deadline + max(1, missed) * period, missed


class WujiHandRuntime:
    def __init__(self, client: WujiHandClient, limiter: SafeCommandLimiter, target_source: Callable[[], Sequence[float] | None], *, rate_hz: float = 30.0, target_filter: OneEuroFilter | None = None, clock=time.monotonic, sleep=time.sleep):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.client, self.limiter, self.target_source = client, limiter, target_source
        self.rate_hz, self.target_filter, self.clock, self.sleep = rate_hz, target_filter, clock, sleep

    def step(self, timestamp: float) -> tuple[float, ...] | None:
        target = self.target_source()
        if target is None:
            return None
        if self.target_filter is not None:
            target = self.target_filter.filter(target, timestamp)
        safe = self.limiter.limit(target, timestamp)
        self.client.write_positions(safe)
        return safe

    def run(self, stop_requested: Callable[[], bool]) -> None:
        period, deadline = 1.0 / self.rate_hz, self.clock()
        while not stop_requested():
            now = self.clock()
            if now < deadline:
                self.sleep(deadline - now)
                now = self.clock()
            self.step(now)
            deadline, _ = advance_periodic_deadline(deadline, period, now)
