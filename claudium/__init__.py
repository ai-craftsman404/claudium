"""Claudium — the agent harness framework for Claude."""

from claudium.core import ClaudiumAgent, ClaudiumSession, ClaudiumTask, init
from claudium.harness.base import HarnessProtocol
from claudium.orchestrator import OrchestratorSession
from claudium.types import (
    ClaudiumConfig,
    ClaudiumEvent,
    ConsensusSignal,
    HarnessResult,
    Role,
    Skill,
    TeamResult,
    TraceRecord,
)

__all__ = [
    "init",
    "ClaudiumAgent",
    "ClaudiumSession",
    "ClaudiumTask",
    "OrchestratorSession",
    "ClaudiumConfig",
    "ClaudiumEvent",
    "HarnessResult",
    "HarnessProtocol",
    "ConsensusSignal",
    "TeamResult",
    "TraceRecord",
    "Role",
    "Skill",
]
