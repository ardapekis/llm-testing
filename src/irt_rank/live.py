"""Provider-neutral cached evaluation with a hard request ceiling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class EvaluationProvider(Protocol):
    def evaluate(self, model_id: str, item_id: str) -> bool: ...


class RequestLimitExceeded(RuntimeError):
    """Raised before a provider call would exceed the configured ceiling."""


class CachedCappedEvaluator:
    def __init__(self, provider: EvaluationProvider, *, request_limit: int) -> None:
        if request_limit < 0:
            raise ValueError("request_limit must be non-negative")
        self._provider = provider
        self._request_limit = request_limit
        self._requests = 0
        self._cache: dict[tuple[str, str], bool] = {}

    @property
    def requests(self) -> int:
        return self._requests

    def evaluate(self, model_id: str, item_id: str) -> bool:
        key = (model_id, item_id)
        if key in self._cache:
            return self._cache[key]
        if self._requests >= self._request_limit:
            raise RequestLimitExceeded("provider request limit reached")
        result = bool(self._provider.evaluate(model_id, item_id))
        self._requests += 1
        self._cache[key] = result
        return result


class FakeProvider:
    """Deterministic in-memory provider used by the reduced M4 integration."""

    def __init__(self, outcomes: Mapping[tuple[str, str], bool]) -> None:
        self._outcomes = dict(outcomes)
        self.calls = 0

    def evaluate(self, model_id: str, item_id: str) -> bool:
        self.calls += 1
        return self._outcomes[(model_id, item_id)]
