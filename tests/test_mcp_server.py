"""Tests for claudium.mcp_server — guarded so they skip when mcp is not installed."""

from __future__ import annotations

import importlib
import sys

import pytest


def _mcp_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


# ── import guard ──────────────────────────────────────────────────────────────

def test_mcp_server_raises_import_error_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    if "claudium.mcp_server" in sys.modules:
        monkeypatch.delitem(sys.modules, "claudium.mcp_server")
    with pytest.raises(ImportError, match="pip install claudium\\[mcp\\]"):
        import claudium.mcp_server  # noqa: F401


# ── tool functions (only run when mcp is installed) ───────────────────────────

@pytest.mark.skipif(not _mcp_available(), reason="mcp package not installed")
@pytest.mark.asyncio
async def test_claudium_list_skills_returns_sorted_list(tmp_path, monkeypatch) -> None:
    from collections.abc import AsyncIterator

    from claudium.core import ClaudiumAgent
    from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult

    class MockHarness:
        async def run(
            self, *, prompt, system_prompt, config, result_tool=None, tools=None
        ) -> HarnessResult:
            return HarnessResult(text="ok")

        async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
            yield ClaudiumEvent(type="text_delta", data={"text": "ok"})

    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "alpha.md").write_text("---\nname: alpha\n---\nAlpha.\n", encoding="utf-8")
    (skills_dir / "beta.md").write_text("---\nname: beta\n---\nBeta.\n", encoding="utf-8")

    from claudium.skills import load_skills
    agent.skills = load_skills(tmp_path)

    import claudium.mcp_server as srv
    monkeypatch.setattr(srv, "_agent", agent)

    result = await srv.claudium_list_skills()
    assert result == ["alpha", "beta"]
