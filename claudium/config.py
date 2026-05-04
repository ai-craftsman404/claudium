"""Configuration loading from claudium.toml."""

from __future__ import annotations

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

from claudium.types import ClaudiumConfig


def load_config(config_path: str | Path | None = None) -> ClaudiumConfig:
    path = Path(config_path) if config_path else _find_config()
    root = path.parent if path and path.exists() else Path.cwd()

    if not path or not path.exists():
        return ClaudiumConfig(root=root)

    with path.open("rb") as f:
        raw = tomllib.load(f)

    agent = raw.get("agent", {})
    sandbox = raw.get("sandbox", {})
    session = raw.get("session", {})
    mcp = raw.get("mcp", {})

    return ClaudiumConfig(
        model=agent.get("model", "claude-opus-4-5"),
        sandbox=agent.get("sandbox", "virtual"),
        root=root,
        allowed_commands=tuple(sandbox.get("allowed_commands", [])),
        allow_compound_commands=sandbox.get("allow_compound_commands", False),
        typed_retries=session.get("typed_retries", 3),
        mcp_servers=mcp.get("servers", []),
    )


def _find_config() -> Path | None:
    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidate = parent / "claudium.toml"
        if candidate.exists():
            return candidate
    return None
