import inspect

import pytest

from axiom.tools.providers import OfflineProvider, ProviderRequest


def test_provider_is_optional_async_and_returns_explicit_untrusted_text():
    # The no-I/O stub completes without suspension. Drive its coroutine directly:
    # Windows' event-loop self-pipe may require prohibited local socket operations
    # in network-denied sandboxes, even when no provider performs network I/O.
    coroutine = OfflineProvider().complete(ProviderRequest(text="Explain this observation"))
    assert inspect.iscoroutine(coroutine)
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    reply = completed.value.value
    assert reply.api_cost_usd == 0
    assert reply.trusted_for_learning is False
    assert "offline" in reply.text.lower()
