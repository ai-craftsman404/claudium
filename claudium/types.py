"""Shared Claudium types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SandboxName = Literal["virtual", "e2b"]
ModelName = Literal[
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str = ""
    instructions: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    path: Path | None = None


@dataclass(frozen=True)
class Role:
    name: str
    instructions: str
    description: str = ""
    model: str | None = None
    path: Path | None = None


@dataclass
class ClaudiumConfig:
    model: str = "claude-opus-4-5"
    sandbox: str = "virtual"
    root: Path = field(default_factory=Path.cwd)
    skills_dir: Path | None = None
    roles_dir: Path | None = None
    agents_dir: Path | None = None
    state_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    allowed_commands: tuple[str, ...] = ()
    allow_compound_commands: bool = False
    typed_retries: int = 3
    mcp_servers: list[str] = field(default_factory=list)


@dataclass
class HarnessResult:
    text: str
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaudiumEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
