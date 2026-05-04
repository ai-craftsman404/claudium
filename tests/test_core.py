"""Integration tests for ClaudiumAgent and ClaudiumSession using MockHarness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from claudium.core import ClaudiumAgent, _result_tool
from claudium.harness.base import HarnessProtocol
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    """Test double for HarnessProtocol — records calls, returns canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock response"]
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    def _next_response(self) -> str:
        text = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return text

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        result_tool: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> HarnessResult:
        self.calls.append({"prompt": prompt, "result_tool": result_tool, "tools": tools})
        return HarnessResult(text=self._next_response())

    async def stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: ClaudiumConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "mock stream"})
        yield ClaudiumEvent(type="message_stop", data={})


assert isinstance(MockHarness(), HarnessProtocol), "MockHarness must satisfy HarnessProtocol"


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    cfg = ClaudiumConfig(root=tmp_path)
    (tmp_path / ".claudium" / "sessions").mkdir(parents=True)
    return cfg


@pytest.fixture
def mock_harness() -> MockHarness:
    return MockHarness()


@pytest.fixture
def agent(config: ClaudiumConfig, mock_harness: MockHarness) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=mock_harness)


async def _session(agent: ClaudiumAgent, session_id: str = "test"):
    return await agent.session(session_id)


# ── prompt ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_returns_harness_text(agent: ClaudiumAgent, mock_harness: MockHarness) -> None:
    session = await _session(agent)
    result = await session.prompt("hello")
    assert result.text == "mock response"


@pytest.mark.asyncio
async def test_prompt_stores_history(agent: ClaudiumAgent, tmp_path: Path) -> None:
    session = await _session(agent)
    await session.prompt("first message")
    messages = await session._messages()
    assert any("first message" in content for _, content in messages)


@pytest.mark.asyncio
async def test_prompt_includes_history_on_second_call(
    agent: ClaudiumAgent, mock_harness: MockHarness
) -> None:
    mock_harness._responses = ["reply one", "reply two"]
    session = await _session(agent)
    await session.prompt("turn one")
    await session.prompt("turn two")
    second_call_prompt = mock_harness.calls[1]["prompt"]
    assert "turn one" in second_call_prompt


# ── typed output ──────────────────────────────────────────────────────────────

class TriageResult(BaseModel):
    severity: str
    labels: list[str]


@pytest.mark.asyncio
async def test_typed_output_parses_json(agent: ClaudiumAgent, mock_harness: MockHarness) -> None:
    mock_harness._responses = ['{"severity": "high", "labels": ["bug"]}']
    session = await _session(agent)
    result = await session.prompt("triage this", result=TriageResult)
    assert isinstance(result, TriageResult)
    assert result.severity == "high"
    assert result.labels == ["bug"]


@pytest.mark.asyncio
async def test_typed_output_injects_result_tool(
    agent: ClaudiumAgent, mock_harness: MockHarness
) -> None:
    mock_harness._responses = ['{"severity": "low", "labels": []}']
    session = await _session(agent)
    await session.prompt("triage this", result=TriageResult)
    assert mock_harness.calls[0]["result_tool"] is not None
    assert mock_harness.calls[0]["result_tool"]["name"] == "structured_result"


@pytest.mark.asyncio
async def test_typed_output_retries_on_invalid_json(
    agent: ClaudiumAgent, mock_harness: MockHarness
) -> None:
    mock_harness._responses = [
        "not json at all",
        '{"severity": "medium", "labels": ["question"]}',
    ]
    agent.config.typed_retries = 1
    session = await _session(agent)
    result = await session.prompt("triage", result=TriageResult)
    assert result.severity == "medium"
    assert len(mock_harness.calls) == 2


@pytest.mark.asyncio
async def test_typed_output_raises_after_max_retries(
    agent: ClaudiumAgent, mock_harness: MockHarness
) -> None:
    mock_harness._responses = ["bad json"] * 5
    agent.config.typed_retries = 2
    session = await _session(agent)
    with pytest.raises(ValueError, match="Structured output failed"):
        await session.prompt("triage", result=TriageResult)
    assert len(mock_harness.calls) == 3  # 1 initial + 2 retries


# ── skill ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_raises_on_unknown(agent: ClaudiumAgent) -> None:
    session = await _session(agent)
    with pytest.raises(KeyError, match="Unknown skill"):
        await session.skill("nonexistent")


@pytest.mark.asyncio
async def test_skill_invokes_prompt_with_rendered_instructions(
    agent: ClaudiumAgent, mock_harness: MockHarness, tmp_path: Path
) -> None:
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "triage.md").write_text(
        "---\nname: triage\n---\nClassify the issue carefully.\n", encoding="utf-8"
    )
    from claudium.skills import load_skills
    agent.skills = load_skills(tmp_path)

    session = await _session(agent)
    await session.skill("triage", args={"issue_number": 99})
    assert "Classify the issue carefully" in mock_harness.calls[0]["prompt"]
    assert "99" in mock_harness.calls[0]["prompt"]


# ── streaming ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_events(agent: ClaudiumAgent) -> None:
    session = await _session(agent)
    events = [e async for e in session.stream("hello")]
    types = [e.type for e in events]
    assert "text_delta" in types
    assert "message_stop" in types


# ── role resolution ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_role_overrides_model(
    agent: ClaudiumAgent, mock_harness: MockHarness, tmp_path: Path
) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "fast.md").write_text(
        "---\nname: fast\nmodel: claude-haiku-4-5\n---\nBe fast.\n", encoding="utf-8"
    )
    from claudium.skills import load_roles
    agent.roles = load_roles(tmp_path)

    session = await _session(agent)
    config_used = session._effective_config(None, "fast")
    assert config_used.model == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_unknown_role_raises(agent: ClaudiumAgent) -> None:
    session = await _session(agent)
    with pytest.raises(KeyError, match="Unknown role"):
        session._effective_role("ghost")


# ── result tool shape ─────────────────────────────────────────────────────────

def test_result_tool_schema_matches_model() -> None:
    tool = _result_tool(TriageResult)
    assert tool["name"] == "structured_result"
    schema = tool["input_schema"]
    assert "severity" in schema["properties"]
    assert "labels" in schema["properties"]
