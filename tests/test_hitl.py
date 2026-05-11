"""TDD tests for v3d Human-in-the-Loop (HITL) — written before implementation."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.teams.session import TeamRunV3Result, TeamSession
from claudium.types import (
    ApprovalCallback,
    ApprovalRequest,
    ApprovalResponse,
    BudgetExceededError,
    ClaudiumConfig,
    ClaudiumEvent,
    HarnessResult,
    SpecialistSummary,
)

# ── MockHarness ───────────────────────────────────────────────────────────────


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def run(self, *, prompt, system_prompt, config, **_) -> HarnessResult:
        if self._responses and self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = "mock"
        self._call_count += 1
        return HarnessResult(text=text)

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "mock"})


# ── Approval callback helpers ─────────────────────────────────────────────────


def make_approval_callback(approved: bool, reason: str | None = None) -> ApprovalCallback:
    """Return an async callback that auto-responds with given approval."""

    async def callback(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(approved=approved, reason=reason)

    return callback


def capturing_callback() -> tuple[ApprovalCallback, list[ApprovalRequest]]:
    """Return (callback, captured_list) — callback stores each ApprovalRequest."""
    captured: list[ApprovalRequest] = []

    async def callback(req: ApprovalRequest) -> ApprovalResponse:
        captured.append(req)
        return ApprovalResponse(approved=True)

    return callback, captured


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    return ClaudiumConfig(root=tmp_path)


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── Good specialist output (passes rule-based adjudication) ───────────────────

_GOOD_OUTPUT = (
    "Transaction TXN-007 $25k payment. SOX control C-3. "
    "High risk anomaly flagged. Per invoice #INV-007."
)


async def _seed_tokens(db_path: Path, input_tokens: int, output_tokens: int) -> None:
    """Insert a call_log row with given token counts to simulate prior consumption."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        await db.execute(
            "insert into call_log"
            "(session_id, skill, model, latency_ms,"
            " input_tokens, output_tokens, success, created_at)"
            " values (?,?,?,?,?,?,?,?)",
            ("hitl-budget", "test", "m", 100.0,
             input_tokens, output_tokens, 1, "2026-05-10T10:00:00+00:00"),
        )
        await db.commit()


# ── Type / dataclass tests ────────────────────────────────────────────────────


def test_specialist_summary_fields() -> None:
    ss = SpecialistSummary(name="x", output="y", fitness_score=0.9)
    assert ss.name == "x"
    assert ss.output == "y"
    assert ss.fitness_score == pytest.approx(0.9)


def test_approval_request_frozen() -> None:
    req = ApprovalRequest(
        run_id="r1",
        session_id="s1",
        domain="legal-compliance",
        prompt="Review contract",
        specialists=[SpecialistSummary(name="clause-extractor", output="ok", fitness_score=0.8)],
        summary="Summary text",
        rule_check_passed=True,
        gaps=[],
        contradictions=[],
        created_at="2026-05-10T10:00:00+00:00",
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        req.domain = "finance-audit"  # type: ignore[misc]


def test_approval_response_approved_true() -> None:
    resp = ApprovalResponse(approved=True)
    assert resp.approved is True


def test_approval_response_reason_optional() -> None:
    resp = ApprovalResponse(approved=False)
    assert resp.reason is None


def test_team_result_stop_reason_none_by_default() -> None:
    result = TeamRunV3Result(
        run_id="r1",
        prompt="p",
        domain="legal-compliance",
        specialist_results=[],
    )
    assert result.stop_reason is None


def test_team_result_truncated_property_budget() -> None:
    result = TeamRunV3Result(
        run_id="r1",
        prompt="p",
        domain="legal-compliance",
        specialist_results=[],
        stop_reason="budget_exceeded",
    )
    assert result.truncated is True


def test_team_result_approval_rejected_property() -> None:
    result = TeamRunV3Result(
        run_id="r1",
        prompt="p",
        domain="legal-compliance",
        specialist_results=[],
        stop_reason="approval_rejected",
    )
    assert result.approval_rejected is True


def test_team_result_both_properties_false_when_no_stop() -> None:
    result = TeamRunV3Result(
        run_id="r1",
        prompt="p",
        domain="legal-compliance",
        specialist_results=[],
        stop_reason=None,
    )
    assert result.truncated is False
    assert result.approval_rejected is False


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_callback_run_proceeds_normally(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    ts = TeamSession(agent=agent, session_id="hitl-no-cb")
    result = await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=None,
    )
    assert result.stop_reason is None


@pytest.mark.asyncio
async def test_callback_receives_approval_request(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    cb, captured = capturing_callback()
    ts = TeamSession(agent=agent, session_id="hitl-capture")
    await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=cb,
        checkpoint="post_specialists",
    )
    assert len(captured) == 1
    req = captured[0]
    assert req.run_id
    assert req.session_id == "hitl-capture"
    assert req.domain == "legal-compliance"
    assert req.prompt == "Review contract"
    assert len(req.specialists) > 0
    assert req.created_at


@pytest.mark.asyncio
async def test_approved_run_completes(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    ts = TeamSession(agent=agent, session_id="hitl-approved")
    result = await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=make_approval_callback(approved=True),
        checkpoint="post_specialists",
    )
    assert result.stop_reason is None
    assert len(result.specialist_results) > 0


@pytest.mark.asyncio
async def test_rejected_run_stops(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    ts = TeamSession(agent=agent, session_id="hitl-rejected")
    result = await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=make_approval_callback(approved=False, reason="Incomplete"),
        checkpoint="post_specialists",
    )
    assert result.stop_reason == "approval_rejected"


@pytest.mark.asyncio
async def test_rejected_result_has_specialist_results(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    ts = TeamSession(agent=agent, session_id="hitl-rej-specs")
    result = await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=make_approval_callback(approved=False, reason="Not enough detail"),
        checkpoint="post_specialists",
    )
    assert len(result.specialist_results) > 0


@pytest.mark.asyncio
async def test_rejected_result_approval_rejected_property(agent: ClaudiumAgent) -> None:
    agent.harness = MockHarness([_GOOD_OUTPUT])
    ts = TeamSession(agent=agent, session_id="hitl-rej-prop")
    result = await ts.run_team_v3(
        "Review contract",
        domain="legal-compliance",
        on_approval_required=make_approval_callback(approved=False),
        checkpoint="post_specialists",
    )
    assert result.approval_rejected is True


@pytest.mark.asyncio
async def test_invalid_checkpoint_raises_value_error(agent: ClaudiumAgent) -> None:
    ts = TeamSession(agent=agent, session_id="hitl-bad-cp")
    with pytest.raises(ValueError, match="checkpoint"):
        await ts.run_team_v3(
            "Review contract",
            domain="legal-compliance",
            on_approval_required=make_approval_callback(approved=True),
            checkpoint="unknown",
        )


@pytest.mark.asyncio
async def test_budget_recheck_on_resume(tmp_path: Path) -> None:
    """Approve at checkpoint but resume into an exhausted budget → truncated or exception."""
    config = ClaudiumConfig(root=tmp_path, token_budget=1, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness([_GOOD_OUTPUT]))
    ts = TeamSession(agent=agent, session_id="hitl-budget")

    # Exhaust the budget before the run begins
    await ts._ensure_store()
    await _seed_tokens(ts.db_path, 1, 0)

    cb = make_approval_callback(approved=True)
    try:
        result = await ts.run_team_v3(
            "Review contract",
            domain="legal-compliance",
            on_approval_required=cb,
            checkpoint="post_specialists",
        )
        # If no exception, result must reflect exhausted budget
        assert result.truncated or result.stop_reason is not None
    except BudgetExceededError:
        pass  # also acceptable — hard stop on budget re-check
