"""Tests for v3a agent teams — domain registry, specialists, fitness scoring, TeamSession."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.teams.domain import DOMAINS, infer_domain, score_fitness
from claudium.teams.specialist import (
    pool_for,
    select_specialists,
)
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._idx = 0

    def _next(self) -> str:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text

    async def run(
        self, *, prompt, system_prompt, config, result_tool=None, tools=None
    ) -> HarnessResult:
        return HarnessResult(text=self._next())

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": self._next()})


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── Domain registry ───────────────────────────────────────────────────────────


def test_domain_registry_has_legal_compliance() -> None:
    assert "legal-compliance" in DOMAINS


def test_legal_compliance_has_task_types() -> None:
    domain = DOMAINS["legal-compliance"]
    assert "clause-extraction" in domain.task_types
    assert "obligation-validation" in domain.task_types
    assert "risk-classification" in domain.task_types


# ── Specialist pool ───────────────────────────────────────────────────────────


def test_pool_for_legal_compliance() -> None:
    pool = pool_for("legal-compliance")
    assert len(pool) == 3
    names = {s.name for s in pool}
    assert "clause-extractor" in names
    assert "obligation-validator" in names
    assert "risk-classifier" in names


def test_pool_for_unknown_domain_returns_empty() -> None:
    assert pool_for("nonexistent") == []


def test_select_specialists_complexity_1() -> None:
    specialists = select_specialists("legal-compliance", complexity=1)
    assert len(specialists) == 1
    assert specialists[0].name == "clause-extractor"


def test_select_specialists_complexity_3() -> None:
    specialists = select_specialists("legal-compliance", complexity=3)
    assert len(specialists) == 3


def test_select_specialists_clamps_to_pool_size() -> None:
    specialists = select_specialists("legal-compliance", complexity=99)
    assert len(specialists) == 3


# ── Fitness scoring ───────────────────────────────────────────────────────────


def test_fitness_score_good_output() -> None:
    text = (
        "The indemnification clause requires the vendor to pay damages. "
        "Risk level: high risk. The parties (buyer and seller) shall comply."
    )
    score = score_fitness(text, "legal-compliance")
    assert score >= 0.75


def test_fitness_score_poor_output() -> None:
    text = "The document looks generally acceptable."
    score = score_fitness(text, "legal-compliance")
    assert score == 0.0


def test_fitness_score_unknown_domain_returns_zero() -> None:
    assert score_fitness("anything", "nonexistent-domain") == 0.0


def test_fitness_score_partial_output() -> None:
    text = "The termination clause binds the parties."
    score = score_fitness(text, "legal-compliance")
    assert 0.0 < score < 1.0


# ── Domain inference ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_infer_domain_legal(config: ClaudiumConfig) -> None:
    harness = MockHarness(["legal-compliance"])
    domain = await infer_domain(
        ["Review this NDA for indemnification clauses", "Extract liability obligations"],
        harness=harness,
        config=config,
    )
    assert domain == "legal-compliance"


@pytest.mark.asyncio
async def test_infer_domain_unknown_falls_back(config: ClaudiumConfig) -> None:
    harness = MockHarness(["some-unrecognised-domain"])
    domain = await infer_domain(
        ["Do something unrelated"],
        harness=harness,
        config=config,
    )
    assert domain == "unknown"


# ── TeamSession ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_team_session_factory(agent: ClaudiumAgent) -> None:
    ts = await agent.team_session("ts-1")
    from claudium.teams.session import TeamSession
    assert isinstance(ts, TeamSession)


@pytest.mark.asyncio
async def test_run_specialists_returns_results(agent: ClaudiumAgent) -> None:
    responses = [
        "Indemnification clause: vendor must pay damages. High risk. Parties: buyer and seller.",
        "Vendor must pay within 30 days. High risk obligation for parties.",
        "Termination clause: high risk. Parties shall give 30 days notice.",
    ]
    agent.harness = MockHarness(responses)
    ts = await agent.team_session("ts-2")
    specialists = select_specialists("legal-compliance", complexity=3)
    results, truncated = await ts.run_specialists(
        "Review this contract", specialists, "legal-compliance"
    )
    assert len(results) == 3
    assert truncated is False
    assert all(r.fitness_score >= 0.0 for r in results)
    assert all(r.fitness_score <= 1.0 for r in results)


@pytest.mark.asyncio
async def test_run_team_v3_end_to_end(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([
        "Indemnification clause found. High risk. Vendor shall pay. Parties: buyer and seller.",
    ])
    ts = await agent.team_session("ts-3")
    result = await ts.run_team_v3(
        "Review the NDA indemnification section",
        domain="legal-compliance",
        complexity=1,
    )
    assert result.domain == "legal-compliance"
    assert len(result.specialist_results) == 1
    assert result.specialist_results[0].specialist.name == "clause-extractor"
    assert result.run_id


@pytest.mark.asyncio
async def test_domain_persisted_to_sqlite(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness(["legal-compliance"])
    ts = await agent.team_session("ts-4")
    await ts.infer_domain(["Review NDA for liability clauses"])
    async with aiosqlite.connect(ts.db_path) as db:
        cursor = await db.execute(
            "SELECT domain_name FROM domain_registry WHERE session_id='ts-4'"
        )
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "legal-compliance"


@pytest.mark.asyncio
async def test_specialist_runs_persisted_to_sqlite(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([
        "The warranty clause requires vendor shall fix. High risk. Parties: buyer and seller."
    ])
    ts = await agent.team_session("ts-5")
    result = await ts.run_team_v3(
        "Review contract", domain="legal-compliance", complexity=1
    )
    async with aiosqlite.connect(ts.db_path) as db:
        cursor = await db.execute(
            "SELECT specialist_name, fitness_score FROM specialist_runs WHERE run_id=?",
            (result.run_id,),
        )
        rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "clause-extractor"
    assert 0.0 <= rows[0][1] <= 1.0


@pytest.mark.asyncio
async def test_run_team_v3_invalid_domain(agent: ClaudiumAgent) -> None:
    ts = await agent.team_session("ts-6")
    with pytest.raises(ValueError, match="Unknown domain"):
        await ts.run_team_v3("prompt", domain="made-up-domain")
