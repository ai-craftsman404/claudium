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


@dataclass(frozen=True)
class ConsensusSignal:
    agreement_score: float       # 0.0 = all disagree, 1.0 = all agree
    majority_output: str | None  # most common response text
    outlier_indices: list[int]   # agent indices that diverged from majority


@dataclass
class TraceRecord:
    session_id: str
    model: str
    latency_ms: float
    skill: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    success: bool = True
    created_at: str = ""


@dataclass
class AgentWeight:
    skill: str
    agent_index: int
    weight: float       # rolling mean of agreement scores, 0.0–1.0
    run_count: int      # number of runs in the rolling window
    updated_at: str


@dataclass
class CalibrationResult:
    skill: str
    samples_run: int
    weights: list[AgentWeight]
    mean_agreement: float


@dataclass
class TeamResult:
    run_id: str                   # UUID, FK to team_runs table
    prompt: str
    outputs: list[HarnessResult]  # one per sub-agent, ordered by agent index
    consensus: ConsensusSignal
    synthesis: str | None = None  # set after orchestrator.synthesise()
    skill: str | None = None
    resolved_at: str | None = None  # "consensus" | "weighted" | "synthesis_needed"
