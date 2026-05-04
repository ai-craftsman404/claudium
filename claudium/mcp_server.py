"""Claudium MCP server — exposes sessions, skills, and tasks as MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "MCP support requires the mcp extra: pip install claudium[mcp]"
    ) from exc

_mcp = FastMCP("Claudium", instructions="Claudium agent harness — run prompts and skills.")
_agent: Any = None


async def _get_agent() -> Any:
    global _agent
    if _agent is None:
        from claudium.core import init
        _agent = await init()
    return _agent


@_mcp.tool()
async def claudium_prompt(
    prompt: str,
    session_id: str = "default",
    model: str | None = None,
) -> str:
    """Send a prompt to a named Claudium session and return the text response."""
    agent = await _get_agent()
    session = await agent.session(session_id)
    result = await session.prompt(prompt, model=model)
    return result.text


@_mcp.tool()
async def claudium_skill(
    skill_name: str,
    args: dict[str, Any] | None = None,
    session_id: str = "default",
) -> str:
    """Invoke a named Claudium skill within a session and return the response."""
    agent = await _get_agent()
    session = await agent.session(session_id)
    result = await session.skill(skill_name, args=args or {})
    return result.text if hasattr(result, "text") else str(result)


@_mcp.tool()
async def claudium_list_skills() -> list[str]:
    """List all skill names available in the current Claudium project."""
    agent = await _get_agent()
    return sorted(agent.skills.keys())


def serve(root: Path | None = None) -> None:
    """Start the Claudium MCP server on stdio (called by `claudium mcp`)."""
    if root:
        import os
        os.chdir(root)
    _mcp.run()
