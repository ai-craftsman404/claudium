"""Extended core tests — session isolation, persistence, secrets, edge cases."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from claudium.core import ClaudiumAgent
from claudium.sandbox.base import SandboxPolicy
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> str:
        text = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return text

    async def run(self, *, prompt, system_prompt, config, result_tool=None, tools=None) -> HarnessResult:
        self.calls.append({"prompt": prompt})
        return HarnessResult(text=self._next())

    async def stream(self, *, prompt, system_prompt, config, tools=None) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "ok"})
        yield ClaudiumEvent(type="message_stop", data={})


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── session isolation ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_sessions_have_separate_history(agent: ClaudiumAgent) -> None:
    s1 = await agent.session("s1")
    s2 = await agent.session("s2")
    await s1.prompt("only in s1")
    s1_messages = await s1._messages()
    s2_messages = await s2._messages()
    assert any("only in s1" in c for _, c in s1_messages)
    assert not any("only in s1" in c for _, c in s2_messages)


@pytest.mark.asyncio
async def test_same_session_id_resumes_history(config: ClaudiumConfig) -> None:
    harness = MockHarness(["first reply", "second reply"])
    agent = ClaudiumAgent(config=config, harness=harness)
    s1 = await agent.session("persistent")
    await s1.prompt("message one")

    s2 = await agent.session("persistent")
    await s2.prompt("message two")
    messages = await s2._messages()
    contents = [c for _, c in messages]
    assert any("message one" in c for c in contents)
    assert any("message two" in c for c in contents)


@pytest.mark.asyncio
async def test_default_session_id_is_stable(agent: ClaudiumAgent) -> None:
    s1 = await agent.session()
    s2 = await agent.session()
    assert s1.session_id == s2.session_id == "default"


# ── session role ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_role_is_set(agent: ClaudiumAgent, tmp_path: Path) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "analyst.md").write_text(
        "---\nname: analyst\n---\nBe precise.\n", encoding="utf-8"
    )
    from claudium.skills import load_roles
    agent.roles = load_roles(tmp_path)
    session = await agent.session("s1", role="analyst")
    assert session.session_role == "analyst"


@pytest.mark.asyncio
async def test_session_role_injected_into_prompt(
    config: ClaudiumConfig, tmp_path: Path
) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "analyst.md").write_text(
        "---\nname: analyst\n---\nBe precise.\n", encoding="utf-8"
    )
    harness = MockHarness()
    agent = ClaudiumAgent(config=config, harness=harness)
    from claudium.skills import load_roles
    agent.roles = load_roles(tmp_path)
    session = await agent.session(role="analyst")
    await session.prompt("hello")
    assert "Be precise" in harness.calls[0]["prompt"]


# ── secrets ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grant_secrets_raises_on_unknown(agent: ClaudiumAgent) -> None:
    session = await agent.session()
    with pytest.raises(KeyError, match="Unknown secret"):
        with session._grant_secrets(["MISSING_SECRET"]):
            pass


@pytest.mark.asyncio
async def test_grant_secrets_does_not_inject_into_prompt(
    config: ClaudiumConfig,
) -> None:
    config.env["MY_SECRET"] = "supersecret"
    harness = MockHarness()
    agent = ClaudiumAgent(config=config, harness=harness)
    session = await agent.session()
    await session.prompt("use the secret", secrets=["MY_SECRET"])
    assert "supersecret" not in harness.calls[0]["prompt"]


# ── harness injection ──────────────────────────────────────────────────────────

def test_agent_uses_injected_harness(config: ClaudiumConfig) -> None:
    harness = MockHarness()
    agent = ClaudiumAgent(config=config, harness=harness)
    assert agent.harness is harness


def test_agent_defaults_to_anthropic_harness(config: ClaudiumConfig) -> None:
    from claudium.harness.anthropic import AnthropicHarness
    agent = ClaudiumAgent(config=config)
    assert isinstance(agent.harness, AnthropicHarness)


# ── init() injection ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_init_accepts_harness_injection(tmp_path: Path) -> None:
    from claudium.core import init
    harness = MockHarness()
    agent = await init(config_path=tmp_path / "claudium.toml", harness=harness)
    assert agent.harness is harness
