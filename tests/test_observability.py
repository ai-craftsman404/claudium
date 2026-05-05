"""Tests for v2b observability — call_log table, latency, token capture."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._idx = 0
        self.calls: list[dict] = []

    def _next(self) -> str:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text

    async def run(
        self, *, prompt, system_prompt, config, result_tool=None, tools=None
    ) -> HarnessResult:
        self.calls.append({"prompt": prompt})
        return HarnessResult(text=self._next())

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": self._next()})


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


@pytest.mark.asyncio
async def test_session_prompt_logs_call(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    await session.prompt("hello")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM call_log")
        (count,) = await cursor.fetchone()  # type: ignore[misc]
    assert count == 1


@pytest.mark.asyncio
async def test_task_prompt_logs_call(agent: ClaudiumAgent) -> None:
    session = await agent.session("s2")
    task = await session.task("t1")
    await task.prompt("hello")
    async with aiosqlite.connect(task.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM call_log")
        (count,) = await cursor.fetchone()  # type: ignore[misc]
    assert count == 1


@pytest.mark.asyncio
async def test_latency_ms_is_positive(agent: ClaudiumAgent) -> None:
    session = await agent.session("s3")
    await session.prompt("hello")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT latency_ms FROM call_log")
        (latency,) = await cursor.fetchone()  # type: ignore[misc]
    assert latency > 0


@pytest.mark.asyncio
async def test_tokens_none_for_mock_harness(agent: ClaudiumAgent) -> None:
    session = await agent.session("s4")
    await session.prompt("hello")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT input_tokens, output_tokens FROM call_log")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None


@pytest.mark.asyncio
async def test_skill_name_logged(agent: ClaudiumAgent, config: ClaudiumConfig) -> None:
    skills_dir = config.root / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "ping.md").write_text("---\nname: ping\n---\nPing.\n", encoding="utf-8")
    from claudium.skills import load_skills
    agent.skills = load_skills(config.root)
    session = await agent.session("s5")
    await session.skill("ping")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT skill FROM call_log")
        (skill_name,) = await cursor.fetchone()  # type: ignore[misc]
    assert skill_name == "ping"


@pytest.mark.asyncio
async def test_synthesise_logs_call(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness(["synth"])
    h0 = MockHarness(["sub"])
    orch = await agent.orchestrator("orch-obs")
    await orch.team(1, harnesses=[h0])
    result = await orch.run_team("q")
    await orch.synthesise(result)
    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute("SELECT skill FROM call_log WHERE skill = 'synthesise'")
        row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_multiple_prompts_log_multiple_rows(agent: ClaudiumAgent) -> None:
    session = await agent.session("s6")
    await session.prompt("one")
    await session.prompt("two")
    await session.prompt("three")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM call_log")
        (count,) = await cursor.fetchone()  # type: ignore[misc]
    assert count == 3
