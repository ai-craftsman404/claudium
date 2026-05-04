"""Harness backend protocol — any harness must satisfy this interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


@runtime_checkable
class HarnessProtocol(Protocol):
    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        result_tool: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> HarnessResult: ...

    def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ClaudiumEvent]: ...
