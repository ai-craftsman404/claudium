"""Tests for ClaudiumTask — child tasks with isolated history and shared sandbox."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from claudium.core import ClaudiumAgent, ClaudiumTask
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def run(self, *, prompt, system_prompt, config, result_tool=None, tools=None) -> HarnessResult:
        self.calls.append({"prompt": prompt})
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return HarnessResult(text=text)

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "ok"})


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── task creation ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_creates_task(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    task = await session.task("t1")
    assert isinstance(task, ClaudiumTask)
    assert task.task_id == "t1"


@pytest.mark.asyncio
async def test_task_generates_id_when_none_given(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    task = await session.task()
    assert task.task_id != ""


# ── isolation ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_history_isolated_from_session(agent: ClaudiumAgent, tmp_path: Path) -> None:
    session = await agent.session("s1")
    task = await session.task("t1")
    await task.prompt("task message")
    session_msgs = await session._messages()
    assert not any("task message" in c for _, c in session_msgs)


@pytest.mark.asyncio
async def test_two_tasks_have_separate_history(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    t1 = await session.task("t1")
    t2 = await session.task("t2")
    await t1.prompt("only in t1")
    t2_path = t2.db_path
    import aiosqlite
    async with aiosqlite.connect(t2_path) as db:
        cursor = await db.execute("select content from messages")
        rows = await cursor.fetchall()
    assert not any("only in t1" in r[0] for r in rows)


# ── shared sandbox ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_shares_sandbox_with_session(agent: ClaudiumAgent) -> None:
    from claudium.sandbox.base import SandboxPolicy
    from claudium.sandbox.virtual import VirtualSandbox
    session = await agent.session("s1")
    task = await session.task("t1")
    assert task.sandbox is session.sandbox


# ── prompt and skill ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_prompt_returns_text(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    task = await session.task("t1")
    result = await task.prompt("do something")
    assert result.text == "mock"


@pytest.mark.asyncio
async def test_task_skill_raises_on_unknown(agent: ClaudiumAgent) -> None:
    session = await agent.session("s1")
    task = await session.task("t1")
    with pytest.raises(KeyError, match="Unknown skill"):
        await task.skill("nonexistent")


@pytest.mark.asyncio
async def test_task_typed_output(agent: ClaudiumAgent) -> None:
    class Result(BaseModel):
        value: int

    harness = MockHarness(['{"value": 99}'])
    agent2 = ClaudiumAgent(config=agent.config, harness=harness)
    session = await agent2.session("s1")
    task = await session.task("t1")
    result = await task.prompt("give me a value", result=Result)
    assert isinstance(result, Result)
    assert result.value == 99


# ── role in task ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_role_injected_into_prompt(agent: ClaudiumAgent, tmp_path: Path) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "analyst.md").write_text(
        "---\nname: analyst\n---\nBe analytical.\n", encoding="utf-8"
    )
    from claudium.skills import load_roles
    harness = MockHarness()
    agent2 = ClaudiumAgent(config=agent.config, harness=harness)
    agent2.roles = load_roles(tmp_path)
    session = await agent2.session("s1")
    task = await session.task("t1", role="analyst")
    await task.prompt("analyse this")
    assert "Be analytical" in harness.calls[0]["prompt"]
