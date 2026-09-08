import time
from collections import deque
from contextlib import asynccontextmanager


class RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limit exceeded")
        self.retry_after = retry_after


class ThreadBusy(Exception):
    pass


class TooManyTurns(Exception):
    pass


class RateLimiter:
    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._last_prune = 0.0

    def check(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._prune(now)
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            raise RateLimited(retry_after=max(0.0, hits[0] + self.window - now))
        hits.append(now)

    def _prune(self, now: float) -> None:
        if now - self._last_prune < self.window:
            return
        self._last_prune = now
        cutoff = now - self.window
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


class TurnGuard:
    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max_concurrent
        self._active: set[str] = set()

    @property
    def active(self) -> int:
        return len(self._active)

    def acquire(self, thread_id: str) -> None:
        if thread_id in self._active:
            raise ThreadBusy(thread_id)
        if len(self._active) >= self.max_concurrent:
            raise TooManyTurns(thread_id)
        self._active.add(thread_id)

    def release(self, thread_id: str) -> None:
        self._active.discard(thread_id)

    @asynccontextmanager
    async def hold(self, thread_id: str):
        self.acquire(thread_id)
        try:
            yield
        finally:
            self.release(thread_id)
