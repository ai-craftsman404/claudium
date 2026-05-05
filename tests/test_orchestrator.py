"""Tests for claudium.orchestrator — OrchestratorSession, agent teams, consensus signals."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent, ClaudiumTask
from claudium.orchestrator import OrchestratorSession, calculate_consensus
from claudium.types import ClaudiumConfig, ClaudiumEvent, ConsensusSignal, HarnessResult, TeamResult

# ── MockHarness ───────────────────────────────────────────────────────────────


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["mock"]
        self._idx = 0
        self.calls: list[dict] = []

    def _next(self) -> str:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text

    async def run(
        self, *, prompt, system_prompt, config, result_tool=None, tools=None
    ) -> HarnessResult:
        self.calls.append({"prompt": prompt, "result_tool": result_tool})
        return HarnessResult(text=self._next())

    async def stream(
        self, *, prompt, system_prompt, config, tools=None
    ) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": self._next()})


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness(["orchestrator mock"]))


# ── calculate_consensus unit tests ────────────────────────────────────────────


def test_calculate_consensus_all_agree() -> None:
    outputs = [HarnessResult(text="yes"), HarnessResult(text="yes"), HarnessResult(text="yes")]
    signal = calculate_consensus(outputs)
    assert signal.agreement_score == pytest.approx(1.0)
    assert signal.majority_output == "yes"
    assert signal.outlier_indices == []


def test_calculate_consensus_majority() -> None:
    outputs = [HarnessResult(text="yes"), HarnessResult(text="yes"), HarnessResult(text="no")]
    signal = calculate_consensus(outputs)
    assert signal.agreement_score == pytest.approx(2 / 3)
    assert signal.majority_output == "yes"
    assert signal.outlier_indices == [2]


def test_calculate_consensus_all_unique() -> None:
    outputs = [HarnessResult(text="a"), HarnessResult(text="b"), HarnessResult(text="c")]
    signal = calculate_consensus(outputs)
    assert signal.agreement_score == pytest.approx(1 / 3)
    assert len(signal.outlier_indices) == 2


def test_calculate_consensus_empty() -> None:
    signal = calculate_consensus([])
    assert signal.agreement_score == 0.0
    assert signal.majority_output is None
    assert signal.outlier_indices == []


def test_calculate_consensus_single() -> None:
    signal = calculate_consensus([HarnessResult(text="solo")])
    assert signal.agreement_score == pytest.approx(1.0)
    assert signal.majority_output == "solo"
    assert signal.outlier_indices == []


# ── OrchestratorSession factory tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_orchestrator_factory(agent: ClaudiumAgent) -> None:
    orch = await agent.orchestrator("orch-1")
    assert isinstance(orch, OrchestratorSession)
    assert orch.session_id == "orch-1"


@pytest.mark.asyncio
async def test_default_orchestrator_session_id(agent: ClaudiumAgent) -> None:
    orch = await agent.orchestrator()
    assert orch.session_id == "orchestrator"


# ── team() tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_team_creates_n_tasks(agent: ClaudiumAgent) -> None:
    orch = await agent.orchestrator("orch-team")
    tasks = await orch.team(3)
    assert len(tasks) == 3
    assert all(isinstance(t, ClaudiumTask) for t in tasks)


@pytest.mark.asyncio
async def test_team_injects_per_agent_harnesses(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["agent-0 answer"])
    h1 = MockHarness(["agent-1 answer"])
    h2 = MockHarness(["agent-2 answer"])
    orch = await agent.orchestrator("orch-inject")
    await orch.team(3, harnesses=[h0, h1, h2])
    assert orch._team[0].agent.harness is h0
    assert orch._team[1].agent.harness is h1
    assert orch._team[2].agent.harness is h2


# ── run_team() tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_team_raises_without_team(agent: ClaudiumAgent) -> None:
    orch = await agent.orchestrator("orch-nogroup")
    with pytest.raises(RuntimeError, match="team()"):
        await orch.run_team("prompt")


@pytest.mark.asyncio
async def test_run_team_calls_all_agents(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["r0"])
    h1 = MockHarness(["r1"])
    h2 = MockHarness(["r2"])
    orch = await agent.orchestrator("orch-calls")
    await orch.team(3, harnesses=[h0, h1, h2])
    await orch.run_team("the prompt")
    assert len(h0.calls) == 1
    assert len(h1.calls) == 1
    assert len(h2.calls) == 1


@pytest.mark.asyncio
async def test_run_team_returns_team_result(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["same answer"])
    h1 = MockHarness(["same answer"])
    h2 = MockHarness(["same answer"])
    orch = await agent.orchestrator("orch-result")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("question")
    assert isinstance(result, TeamResult)
    assert isinstance(result.consensus, ConsensusSignal)
    assert len(result.outputs) == 3


@pytest.mark.asyncio
async def test_run_team_returns_full_consensus(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["same answer"])
    h1 = MockHarness(["same answer"])
    h2 = MockHarness(["same answer"])
    orch = await agent.orchestrator("orch-full")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("question")
    assert result.consensus.agreement_score == pytest.approx(1.0)
    assert result.consensus.outlier_indices == []


@pytest.mark.asyncio
async def test_run_team_detects_outlier(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["majority"])
    h1 = MockHarness(["majority"])
    h2 = MockHarness(["outlier response"])
    orch = await agent.orchestrator("orch-outlier")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("question")
    assert result.consensus.outlier_indices == [2]
    assert result.consensus.agreement_score == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_run_team_records_to_sqlite(agent: ClaudiumAgent) -> None:
    h0 = MockHarness(["r0"])
    h1 = MockHarness(["r1"])
    h2 = MockHarness(["r2"])
    orch = await agent.orchestrator("orch-sqlite")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("stored prompt")

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute("select count(*) from team_runs")
        (run_count,) = await cursor.fetchone()  # type: ignore[misc]
        cursor = await db.execute("select count(*) from agent_outputs")
        (output_count,) = await cursor.fetchone()  # type: ignore[misc]

    assert run_count == 1
    assert output_count == 3
    assert result.run_id  # non-empty UUID


# ── synthesise() tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesise_calls_orchestrator_harness(agent: ClaudiumAgent) -> None:
    orch_harness = MockHarness(["synthesised answer"])
    agent.harness = orch_harness
    h0 = MockHarness(["sub output"])
    h1 = MockHarness(["sub output"])
    h2 = MockHarness(["sub output"])
    orch = await agent.orchestrator("orch-synth")
    await orch.team(3, harnesses=[h0, h1, h2])
    result = await orch.run_team("question")
    synthesis = await orch.synthesise(result)
    assert synthesis == "synthesised answer"
    assert len(orch_harness.calls) == 1


@pytest.mark.asyncio
async def test_synthesise_patches_result_synthesis(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness(["best answer"])
    h0 = MockHarness(["answer"])
    orch = await agent.orchestrator("orch-patch")
    await orch.team(1, harnesses=[h0])
    result = await orch.run_team("q")
    assert result.synthesis is None
    await orch.synthesise(result)
    assert result.synthesis == "best answer"


@pytest.mark.asyncio
async def test_synthesise_updates_sqlite(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness(["final answer"])
    h0 = MockHarness(["sub"])
    h1 = MockHarness(["sub"])
    orch = await agent.orchestrator("orch-update")
    await orch.team(2, harnesses=[h0, h1])
    result = await orch.run_team("q")
    await orch.synthesise(result)

    async with aiosqlite.connect(orch.db_path) as db:
        cursor = await db.execute(
            "select synthesis from team_runs where id = ?", (result.run_id,)
        )
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "final answer"


# ── Integration test (one real API call — skipped in CI) ─────────────────────


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipped in CI",
)
@pytest.mark.asyncio
async def test_synthesise_with_real_api(tmp_path: Path) -> None:
    from claudium.harness.anthropic import AnthropicHarness

    sub_h0 = MockHarness(["Paris is the capital of France."])
    sub_h1 = MockHarness(["The capital of France is Paris."])
    sub_h2 = MockHarness(["Paris."])

    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=AnthropicHarness())
    orch = await agent.orchestrator("integration-test")
    await orch.team(3, harnesses=[sub_h0, sub_h1, sub_h2])
    result = await orch.run_team("What is the capital of France?")
    synthesis = await orch.synthesise(result)
    assert "Paris" in synthesis
    assert result.synthesis is not None
