from __future__ import annotations

import pytest

from irt_rank.live import CachedCappedEvaluator, FakeProvider, RequestLimitExceeded


def test_cached_evaluator_enforces_cap_before_provider_call() -> None:
    provider = FakeProvider({("m1", "i1"): True, ("m2", "i1"): False})
    evaluator = CachedCappedEvaluator(provider, request_limit=1)
    assert evaluator.evaluate("m1", "i1")
    assert evaluator.evaluate("m1", "i1")
    assert provider.calls == 1
    with pytest.raises(RequestLimitExceeded):
        evaluator.evaluate("m2", "i1")
    assert provider.calls == 1
