"""Claudium agent teams — v3 specialist team orchestration."""

from claudium.teams.domain import DOMAINS, Domain, infer_domain, score_fitness
from claudium.teams.session import (
    AdjudicationResult,
    SpecialistResult,
    TeamRunV3Result,
    TeamSession,
)
from claudium.teams.specialist import Specialist, pool_for, select_specialists

__all__ = [
    "DOMAINS",
    "Domain",
    "infer_domain",
    "score_fitness",
    "Specialist",
    "pool_for",
    "select_specialists",
    "AdjudicationResult",
    "SpecialistResult",
    "TeamRunV3Result",
    "TeamSession",
]
