"""Tests for v3b — Finance/Audit domain, hybrid adjudication, ReplayHarness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.harness.replay import ReplayHarness
from claudium.teams.domain import DOMAINS, score_fitness
from claudium.teams.session import (
    SpecialistResult,
    _adjudicate_rule_based,
    _infer_complexity,
)
from claudium.teams.specialist import Specialist, pool_for
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


# ── Finance/Audit domain ──────────────────────────────────────────────────────


def test_finance_audit_domain_in_registry() -> None:
    assert "finance-audit" in DOMAINS


def test_finance_audit_has_sequential_strategy() -> None:
    assert DOMAINS["finance-audit"].execution_strategy == "sequential"


def test_legal_compliance_has_parallel_strategy() -> None:
    assert DOMAINS["legal-compliance"].execution_strategy == "parallel"


def test_finance_audit_specialist_pool() -> None:
    pool = pool_for("finance-audit")
    assert len(pool) == 3
    names = {s.name for s in pool}
    assert "transaction-auditor" in names
    assert "risk-analyst" in names
    assert "compliance-checker" in names


def test_finance_audit_fitness_good_output() -> None:
    text = (
        "Transaction #TXN-4421 amount $52,000 payment to vendor. "
        "Anomaly: threshold breach. SOX control ref: C-12. "
        "Based on invoice #INV-991, high risk flagged."
    )
    score = score_fitness(text, "finance-audit")
    assert score >= 0.75


def test_finance_audit_fitness_poor_output() -> None:
    score = score_fitness("The financials look broadly acceptable.", "finance-audit")
    assert score == 0.0


def test_finance_audit_fitness_evidence_cited() -> None:
    text = "Per transaction id TXN-001: AML suspicious pattern. SOX breach. High risk."
    score = score_fitness(text, "finance-audit")
    assert score == 1.0


# ── Complexity inference ──────────────────────────────────────────────────────


def test_complexity_short_prompt() -> None:
    assert _infer_complexity("Review this invoice.") == 1


def test_complexity_medium_prompt() -> None:
    prompt = " ".join(["word"] * 80)
    assert _infer_complexity(prompt) == 2


def test_complexity_long_prompt() -> None:
    prompt = " ".join(["word"] * 200)
    assert _infer_complexity(prompt) == 3


# ── Rule-based adjudication ───────────────────────────────────────────────────


def _make_sr(name: str, domain: str, text: str) -> SpecialistResult:
    spec = Specialist(name=name, domain=domain, focus="", instructions="")
    fitness = score_fitness(text, domain)
    return SpecialistResult(specialist=spec, output=HarnessResult(text=text), fitness_score=fitness)


def test_rule_based_accepts_good_outputs() -> None:
    sr = _make_sr(
        "transaction-auditor", "finance-audit",
        "Transaction TXN-001 $10k payment. SOX control C-1. High risk anomaly. Per invoice #99.",
    )
    adj = _adjudicate_rule_based([sr], "finance-audit")
    assert adj.mode == "rule-based"
    assert adj.accepted


def test_rule_based_rejects_low_fitness() -> None:
    sr = _make_sr("transaction-auditor", "finance-audit", "Looks fine.")
    adj = _adjudicate_rule_based([sr], "finance-audit")
    assert not adj.accepted
    assert len(adj.gaps) > 0
    assert "transaction-auditor" in adj.re_dispatch


def test_rule_based_detects_contradiction() -> None:
    sr1 = _make_sr(
        "risk-analyst", "finance-audit",
        "Transaction TXN-1. SOX. Critical risk anomaly. Per ref #1.",
    )
    sr2 = _make_sr(
        "compliance-checker", "finance-audit",
        "Transaction TXN-1. AML. Low risk. Based on invoice #1.",
    )
    adj = _adjudicate_rule_based([sr1, sr2], "finance-audit")
    assert not adj.accepted
    assert len(adj.contradictions) > 0


# ── Hybrid adjudication in TeamSession ───────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_adjudication_triggered_on_low_fitness(agent: ClaudiumAgent) -> None:
    responses = [
        "Looks fine.",           # transaction-auditor — low fitness → triggers LLM
        "synthesised finding",   # LLM adjudication response
    ]
    agent.harness = MockHarness(responses)
    ts = await agent.team_session("ts-adj-1")
    result = await ts.run_team_v3("Review invoice", domain="finance-audit", complexity=1)
    assert result.adjudication is not None
    assert result.adjudication.mode == "llm"
    assert result.synthesis == "synthesised finding"


@pytest.mark.asyncio
async def test_llm_adjudication_not_triggered_on_good_output(agent: ClaudiumAgent) -> None:
    good = (
        "Transaction TXN-007 $25k. SOX control C-3. "
        "High risk anomaly flagged. Per invoice #INV-007."
    )
    agent.harness = MockHarness([good])
    ts = await agent.team_session("ts-adj-2")
    result = await ts.run_team_v3("Review invoice", domain="finance-audit", complexity=1)
    assert result.adjudication is not None
    assert result.adjudication.mode == "rule-based"
    assert result.adjudication.accepted
    assert result.synthesis is None


# ── Sequential execution ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sequential_passes_prior_findings(agent: ClaudiumAgent) -> None:
    prompts_seen: list[str] = []

    async def capturing_run(*, prompt, **kwargs) -> HarnessResult:
        prompts_seen.append(prompt)
        return HarnessResult(
            text="Transaction TXN-1. SOX. High risk. Per ref #1. Based on invoice #1."
        )

    agent.harness.run = capturing_run  # type: ignore[method-assign]
    ts = await agent.team_session("ts-seq-1")
    await ts.run_team_v3("Review transaction", domain="finance-audit", complexity=2)

    # Second specialist prompt should contain first specialist's findings
    assert len(prompts_seen) >= 2
    assert "findings" in prompts_seen[1].lower()


# ── ReplayHarness ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_harness_records_to_sqlite(tmp_path: Path, config: ClaudiumConfig) -> None:
    db = tmp_path / "replay.db"
    backing = MockHarness(["recorded response"])
    harness = ReplayHarness(db, record=True, backing_harness=backing)
    result = await harness.run(prompt="test prompt", system_prompt="", config=config)
    assert result.text == "recorded response"
    async with aiosqlite.connect(db) as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM replay_log")
        (count,) = await cursor.fetchone()  # type: ignore[misc]
    assert count == 1


@pytest.mark.asyncio
async def test_replay_harness_replays_deterministically(
    tmp_path: Path, config: ClaudiumConfig
) -> None:
    db = tmp_path / "replay.db"
    backing = MockHarness(["original response"])
    record_harness = ReplayHarness(db, record=True, backing_harness=backing)
    await record_harness.run(prompt="audit prompt", system_prompt="", config=config)

    replay_harness = ReplayHarness(db, record=False)
    result = await replay_harness.run(prompt="audit prompt", system_prompt="", config=config)
    assert result.text == "original response"


@pytest.mark.asyncio
async def test_replay_harness_raises_on_missing_fixture(
    tmp_path: Path, config: ClaudiumConfig
) -> None:
    db = tmp_path / "replay.db"
    harness = ReplayHarness(db, record=False)
    with pytest.raises(KeyError, match="No recorded response"):
        await harness.run(prompt="unseen prompt", system_prompt="", config=config)


@pytest.mark.asyncio
async def test_replay_harness_same_prompt_same_output(
    tmp_path: Path, config: ClaudiumConfig
) -> None:
    db = tmp_path / "replay.db"
    backing = MockHarness(["deterministic output"])
    await ReplayHarness(db, record=True, backing_harness=backing).run(
        prompt="p", system_prompt="", config=config
    )
    r1 = await ReplayHarness(db, record=False).run(prompt="p", system_prompt="", config=config)
    r2 = await ReplayHarness(db, record=False).run(prompt="p", system_prompt="", config=config)
    assert r1.text == r2.text == "deterministic output"
