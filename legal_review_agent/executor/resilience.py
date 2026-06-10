"""基建级容错原语：限流、熔断。供工具执行网关使用，不含任何业务/认知逻辑。"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """简单令牌桶：按每分钟配额限流，超额时阻塞等待。"""

    def __init__(self, per_minute: int):
        self._capacity = per_minute
        self._tokens = float(per_minute)
        self._rate = per_minute / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self._rate
            time.sleep(wait)


class CircuitBreaker:
    """按工具维度熔断：连续失败超过阈值后打开，冷却期后半开试探。"""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int, recovery_time_s: float):
        self._threshold = failure_threshold
        self._recovery = recovery_time_s
        self._failures = 0
        self._state = self.CLOSED
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allow(self) -> bool:
        with self._lock:
            self._maybe_half_open()
            return self._state != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def _maybe_half_open(self) -> None:
        if self._state == self.OPEN and time.monotonic() - self._opened_at >= self._recovery:
            self._state = self.HALF_OPEN
