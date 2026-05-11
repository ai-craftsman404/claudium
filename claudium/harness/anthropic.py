"""Anthropic SDK harness — native tool-use, streaming, prompt caching."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class AnthropicHarness:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or anthropic.AsyncAnthropic()

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        result_tool: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> HarnessResult:
        all_tools = list(tools or [])
        tool_choice: dict[str, Any] | anthropic.NotGiven = anthropic.NOT_GIVEN

        if result_tool:
            all_tools.append(result_tool)
            tool_choice = {"type": "tool", "name": result_tool["name"]}

        kwargs: dict[str, Any] = {
            "model": config.model or "claude-opus-4-5",
            "max_tokens": 4096,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if system_prompt
            else anthropic.NOT_GIVEN,
            "messages": [{"role": "user", "content": prompt}],
        }
        if all_tools:
            kwargs["tools"] = all_tools
            kwargs["tool_choice"] = tool_choice

        response = await self._client.messages.create(**kwargs)

        if result_tool:
            for block in response.content:
                if block.type == "tool_use" and block.name == result_tool["name"]:
                    return HarnessResult(
                        text=json.dumps(block.input),
                        model=response.model,
                        raw=response,
                        metadata={"tool_use": True},
                    )

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        )
        return HarnessResult(text=text, model=response.model, raw=response)

    async def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ClaudiumEvent]:
        async with self._client.messages.stream(
            model=config.model or "claude-opus-4-5",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if system_prompt
            else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
            tools=tools or anthropic.NOT_GIVEN,
        ) as stream:
            async for text in stream.text_stream:
                yield ClaudiumEvent(type="text_delta", data={"text": text})
            final = await stream.get_final_message()
            yield ClaudiumEvent(
                type="message_stop",
                data={"usage": final.usage.model_dump() if final.usage else {}},
            )
