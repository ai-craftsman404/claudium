"""Tests for v2c self-improvement — routing weights, evaluation tree, calibration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.types import CalibrationResult, ClaudiumConfig, ClaudiumEvent, HarnessResult


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._idx = 0
        self.call_count = 0

    def _next(self) -> str:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text

    async def run(
        self, *, prompt, system_prompt, config, result_tool=None, tools=None
    ) -> HarnessResult:
        self.call_count += 1
        return HarnessResult(text=self._next())

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": self._next()})


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── Weight initialisation ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weights_initialise_neutral(agent: ClaudiumAgent) -> None:
    """Fresh orchestrator returns weight=1.0 for all agents — neutral prior."""
    from claudium.orchestrator import _get_weights

    orch = await agent.orchestrator("orch-init")
    await orch._ensure_store()
    weights = await _get_weights(orch.db_path, "triage", 3)
    assert weights == [1.0, 1.0, 1.0]


# ── Weight updates ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weight_updates_after_run_team(agent: ClaudiumAgent) -> None:
    """After a run with an outlier agent, the outlier's weight drops below 1.0."""
    h0 = MockHarness(["answer-A"])
    h1 = MockHarness(["answer-A"])
    h2 = MockHarness(["answer-B"])  # outlier
    orch = await agent.orchestrator("orch-weights")
    await orch.team(3, harnesses=[h0, h1, h2])
    await orch.run_team("prompt", skill="triage")

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute(
            "SELECT agent_index, weight FROM agent_weights"
            " WHERE skill='triage' ORDER BY agent_index"
        )
        rows = await cursor.fetchall()

    weight_map = {int(r[0]): float(r[1]) for r in rows}
    assert weight_map[0] == 1.0   # agreed
    assert weight_map[1] == 1.0   # agreed
    assert weight_map[2] == 0.0   # outlier — never agreed


@pytest.mark.asyncio
async def test_weight_rolling_window(agent: ClaudiumAgent) -> None:
    """After window runs, weights converge within [0.0, 1.0] and reflect recent history.

    Three agents — agent 0 agrees in runs 1-2 then diverges in runs 3-4.
    Agents 1 and 2 always agree. Three agents guarantee a clear majority in all runs
    (no tie-break ambiguity from Counter insertion order).
    """
    window = 4
    # Runs 1-2: all agree on "A". Runs 3-4: agent0="B", agents1+2="A" → clear majority "A".
    h0_responses = ["A", "A", "B", "B"]
    h1_responses = ["A", "A", "A", "A"]  # always in majority
    h2_responses = ["A", "A", "A", "A"]  # always in majority

    orch = await agent.orchestrator("orch-window", weight_window=window)
    for run_idx in range(window):
        h0 = MockHarness([h0_responses[run_idx]])
        h1 = MockHarness([h1_responses[run_idx]])
        h2 = MockHarness([h2_responses[run_idx]])
        await orch.team(3, harnesses=[h0, h1, h2])
        await orch.run_team(f"prompt-{run_idx}", skill="rolling")

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute(
            "SELECT agent_index, weight, run_count FROM agent_weights WHERE skill='rolling'"
        )
        rows = await cursor.fetchall()

    weight_map = {int(r[0]): (float(r[1]), int(r[2])) for r in rows}
    w0, c0 = weight_map[0]
    w1, c1 = weight_map[1]

    assert c0 == window
    assert c1 == window
    assert 0.0 <= w0 <= 1.0
    assert w1 == 1.0  # always agreed — weight stays 1.0
    assert w0 < w1    # agent 0 diverged in later runs → lower weight


# ── Evaluation tree routing ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_high_consensus_skips_synthesis(agent: ClaudiumAgent) -> None:
    """When agreement_score >= high_threshold, synthesise() is never invoked."""
    h0 = MockHarness(["same answer"])
    h1 = MockHarness(["same answer"])
    h2 = MockHarness(["same answer"])

    orch = await agent.orchestrator("orch-high")
    await orch.team(3, harnesses=[h0, h1, h2])

    synth_called = False
    original_synthesise = orch.synthesise

    async def mock_synthesise(result, **kwargs):  # type: ignore[override]
        nonlocal synth_called
        synth_called = True
        return await original_synthesise(result, **kwargs)

    orch.synthesise = mock_synthesise  # type: ignore[method-assign]
    result = await orch.run_team("q", auto_synthesise=True)

    assert result.consensus.agreement_score == 1.0
    assert result.resolved_at == "consensus"
    assert not synth_called


@pytest.mark.asyncio
async def test_low_consensus_triggers_synthesis(agent: ClaudiumAgent) -> None:
    """When agreement is below both thresholds, auto_synthesise triggers synthesise()."""
    h0 = MockHarness(["alpha"])
    h1 = MockHarness(["beta"])
    h2 = MockHarness(["gamma"])

    agent.harness = MockHarness(["synthesised result"])
    orch = await agent.orchestrator("orch-low")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("q", auto_synthesise=True)

    assert result.resolved_at == "synthesis_needed"
    assert result.synthesis == "synthesised result"


# ── Calibration ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_calibrate_returns_result(agent: ClaudiumAgent) -> None:
    """calibrate() returns a CalibrationResult with correct sample count."""
    h0 = MockHarness(["X", "X", "Y"])
    h1 = MockHarness(["X", "X", "Y"])

    orch = await agent.orchestrator("orch-cal")
    await orch.team(2, harnesses=[h0, h1])
    cal = await orch.calibrate("triage", ["s1", "s2", "s3"])

    assert isinstance(cal, CalibrationResult)
    assert cal.skill == "triage"
    assert cal.samples_run == 3
    assert 0.0 <= cal.mean_agreement <= 1.0


@pytest.mark.asyncio
async def test_calibrate_updates_weights(agent: ClaudiumAgent) -> None:
    """After calibration, agent_weights table is populated (non-empty)."""
    h0 = MockHarness(["ans"] * 5)
    h1 = MockHarness(["ans"] * 5)

    orch = await agent.orchestrator("orch-cal-weights")
    await orch.team(2, harnesses=[h0, h1])
    await orch.calibrate("perf", ["p1", "p2", "p3"])

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM agent_weights WHERE skill='perf'")
        (count,) = await cursor.fetchone()  # type: ignore[misc]

    assert count == 2  # one row per agent


# ── SQLite persistence ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weights_persisted_to_sqlite(agent: ClaudiumAgent) -> None:
    """agent_weights table is populated after a run_team call with a skill."""
    h0 = MockHarness(["yes"])
    h1 = MockHarness(["no"])

    orch = await agent.orchestrator("orch-persist")
    await orch.team(2, harnesses=[h0, h1])
    await orch.run_team("persist?", skill="code-review")

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute(
            "SELECT agent_index, weight FROM agent_weights WHERE skill='code-review'"
        )
        rows = await cursor.fetchall()

    assert len(rows) == 2
    indices = {int(r[0]) for r in rows}
    assert indices == {0, 1}
