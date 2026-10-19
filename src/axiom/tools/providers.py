from typing import Literal, Protocol

from pydantic import Field

from axiom.core.models import Record


class ProviderRequest(Record):
    text: str
    max_tokens: int = Field(default=256, ge=1, le=4096)


class ProviderReply(Record):
    text: str
    provider: str
    api_cost_usd: float = Field(default=0, ge=0)
    trusted_for_learning: Literal[False] = False


class LLMProvider(Protocol):
    async def complete(self, request: ProviderRequest) -> ProviderReply: ...


class OfflineProvider:
    async def complete(self, request: ProviderRequest) -> ProviderReply:
        return ProviderReply(text="Offline provider: language generation is disabled.", provider="offline")
