"""Integration workflow tests for token budget enforcement in sequential specialist execution.

This file tests the SEQUENTIAL execution path used by the finance-audit domain.
It verifies that _check_budget() is called between specialists and that the
(results, truncated) tuple is correctly populated when budget is exhausted mid-run.

Critical architectural gap being tested:
  - ClaudiumTask writes tokens to its own DB (task-{id}.db), NOT the session DB
  - _check_budget() reads the session DB, so specialist token usage is invisible
  - All tests work by seeding tokens directly into the session DB

Test scenarios:
  A. Budget exceeded BEFORE any specialists run → truncated=True, specialist_results == []
  B. Budget passes initial check, re-check fires before 2nd specialist → truncated=True, len(specialist_results) == 1
  C. Full sequential run within budget → truncated=False, all specialists return results
  D. Document the task-token-visibility gap (xfail if budget somehow catches it)
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.types import BudgetExceededError, ClaudiumConfig, ClaudiumEvent, HarnessResult


# ── Mock Harness ──────────────────────────────────────────────────────────────


class MockHarness:
    """Mock harness that optionally writes tokens to a specific DB path on each call."""

    def __init__(
        self,
        responses: list[str] | None = None,
        token_writer_session_db: Path | None = None,
        tokens_per_call: tuple[int, int] = (100, 50),
    ) -> None:
        """
        Args:
            responses: List of responses to return on successive calls.
            token_writer_session_db: If provided, write tokens to this DB after each call.
                                     This simulates session-level adjudication calls.
            tokens_per_call: (input_tokens, output_tokens) to write if token_writer_session_db is set.
        """
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self._token_writer_session_db = token_writer_session_db
        self._tokens_per_call = tokens_per_call

    async def run(self, *, prompt, system_prompt, config, **_) -> HarnessResult:
        if self._responses and self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = "mock response"
        self._call_count += 1

        # If token_writer is configured, write tokens to session DB after this call
        if self._token_writer_session_db:
            input_tok, output_tok = self._tokens_per_call
            async with aiosqlite.connect(self._token_writer_session_db) as db:
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
                    (
                        "sequential-session",
                        "test",
                        "m",
                        100.0,
                        input_tok,
                        output_tok,
                        1,
                        "2026-05-10T10:00:00+00:00",
                    ),
                )
                await db.commit()

        return HarnessResult(text=text)

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "mock"})


# ── Helper: seed tokens into session DB ───────────────────────────────────────


async def _seed_tokens(db_path: Path, input_tokens: int, output_tokens: int) -> None:
    """Insert a call_log row with the given token counts into the session DB."""
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
            (
                "sequential-session",
                "test",
                "m",
                100.0,
                input_tokens,
                output_tokens,
                1,
                "2026-05-10T10:00:00+00:00",
            ),
        )
        await db.commit()


# ── Helper: verify finance-audit uses sequential execution ────────────────────


def test_finance_audit_domain_uses_sequential_strategy() -> None:
    """Confirm that finance-audit domain has execution_strategy='sequential'."""
    from claudium.teams.domain import DOMAINS

    finance_audit = DOMAINS.get("finance-audit")
    assert finance_audit is not None, "finance-audit domain not found"
    assert (
        finance_audit.execution_strategy == "sequential"
    ), f"Expected 'sequential', got '{finance_audit.execution_strategy}'"


# ── Test A: Budget exceeded BEFORE any specialists run ────────────────────────


@pytest.mark.asyncio
async def test_sequential_budget_exceeded_before_specialists() -> None:
    """Test A: Budget exceeded at first check → truncated=True, specialist_results == []

    This test seeds the session DB with tokens at the limit before calling run_team_v3.
    The first _check_budget() call in run_team_v3 should raise BudgetExceededError,
    which is caught and sets truncated=True without running any specialists.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = ClaudiumConfig(
            root=tmp_path, token_budget=1000, budget_grace_pct=0.0
        )
        agent = ClaudiumAgent(config=config, harness=MockHarness())

        from claudium.teams.session import TeamSession

        ts = TeamSession(agent=agent, session_id="seq-budget-before")
        await ts._ensure_store()

        # Seed session DB with tokens at the limit
        await _seed_tokens(ts.db_path, 1000, 0)  # exactly at limit

        # Run team — first _check_budget() should raise
        result = await ts.run_team_v3(
            "Review invoice", domain="finance-audit", complexity=2
        )

        # Assertions
        assert result.truncated is True, "Expected truncated=True when budget exceeded before specialists"
        assert (
            len(result.specialist_results) == 0
        ), "Expected no specialist results when budget hit at initial check"


# ── Test B: Budget passes initial check, re-check fires before 2nd specialist ──


@pytest.mark.asyncio
async def test_sequential_budget_exceeded_between_specialists() -> None:
    """Test B: Budget passes initial check, then re-check fires before 2nd specialist.

    This test:
    1. Seeds session DB with 0 tokens initially
    2. Uses a MockHarness configured to write tokens to the SESSION DB after each call
    3. Runs run_team_v3 with complexity=2 (which requests 2 specialists)
    4. After the 1st specialist completes, the MockHarness has written tokens to the session DB
    5. The 2nd specialist's pre-check (_check_budget()) sees those tokens and raises
    6. Expected: truncated=True, len(specialist_results) == 1

    This simulates what would happen if session-level adjudication calls accumulated tokens.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Budget of 150 tokens, no grace
        config = ClaudiumConfig(
            root=tmp_path, token_budget=150, budget_grace_pct=0.0
        )
        agent = ClaudiumAgent(
            config=config,
            harness=MockHarness(
                responses=["specialist-1-output", "specialist-2-output"],
                token_writer_session_db=tmp_path / "seq-budget-between.db",
                tokens_per_call=(100, 100),  # Each specialist call writes 200 tokens
            ),
        )

        from claudium.teams.session import TeamSession

        ts = TeamSession(agent=agent, session_id="seq-budget-between")
        ts.db_path = tmp_path / "seq-budget-between.db"
        await ts._ensure_store()

        # Start with 0 tokens in the session
        # (MockHarness will write tokens after 1st specialist)

        # Run team with complexity=2 to request 2 specialists
        result = await ts.run_team_v3(
            "Review audit findings", domain="finance-audit", complexity=2
        )

        # Assertions
        assert (
            result.truncated is True
        ), "Expected truncated=True when budget exceeded between specialists"
        assert (
            len(result.specialist_results) == 1
        ), f"Expected exactly 1 specialist result, got {len(result.specialist_results)}"
        assert (
            result.specialist_results[0].specialist.name is not None
        ), "First specialist should have a name"


# ── Test C: Full sequential run within budget ─────────────────────────────────


@pytest.mark.asyncio
async def test_sequential_full_run_within_budget() -> None:
    """Test C: Full sequential run within budget → truncated=False, all specialists complete.

    This test uses responses that satisfy the finance-audit domain's fitness criteria
    so that adjudication passes and all specialists are included in the result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Large budget to allow full execution
        config = ClaudiumConfig(root=tmp_path, token_budget=100_000)

        # Responses that satisfy fitness criteria for finance-audit domain
        good_response = (
            "Transaction TXN-001 amount $50000. SOX control C-12. "
            "High risk anomaly flagged. Per invoice #INV-001."
        )

        agent = ClaudiumAgent(
            config=config, harness=MockHarness(responses=[good_response, good_response])
        )

        from claudium.teams.session import TeamSession

        ts = TeamSession(agent=agent, session_id="seq-within-budget")
        await ts._ensure_store()

        # Run team with complexity=2 to request 2 specialists
        result = await ts.run_team_v3(
            "Review financial transactions", domain="finance-audit", complexity=2
        )

        # Assertions
        assert (
            result.truncated is False
        ), "Expected truncated=False when within budget"
        assert (
            len(result.specialist_results) == 2
        ), f"Expected 2 specialist results, got {len(result.specialist_results)}"
        # Both should be from finance-audit domain
        for sr in result.specialist_results:
            assert sr.specialist.domain == "finance-audit"


# ── Test D: Document the task-token-visibility gap ────────────────────────────


@pytest.mark.asyncio
async def test_sequential_task_tokens_invisible_to_budget() -> None:
    """Test D: Document that task-token writes are invisible to budget enforcement.

    This test demonstrates the known architectural gap:
    - ClaudiumTask writes tokens to task-{id}.db (child process)
    - _check_budget() reads the session db only
    - Therefore, specialist token usage is NOT counted in budget checks

    In a real scenario, if specialists consumed significant tokens via API calls,
    those tokens would be written to task-{id}.db, not the session DB.
    Thus, _check_budget() would see nothing and not raise, even if specialists
    exceeded the budget.

    This test is marked xfail if the budget somehow catches the gap (unexpected),
    or passes if it demonstrates the gap (expected behavior, documenting the issue).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Very tight budget: 100 tokens
        config = ClaudiumConfig(
            root=tmp_path, token_budget=100, budget_grace_pct=0.0
        )

        # Harness that returns responses (but token writes to task DB, not session DB)
        agent = ClaudiumAgent(
            config=config,
            harness=MockHarness(
                responses=["result-1", "result-2"],
                # Critically: NOT writing tokens to session DB
                # In reality, specialists' task.prompt() calls would log to task-{id}.db
            ),
        )

        from claudium.teams.session import TeamSession

        ts = TeamSession(agent=agent, session_id="seq-task-invisible")
        await ts._ensure_store()

        # Start with 0 tokens in the session DB
        # Specialists will "consume" tokens (in task DB) but they won't be visible

        # Run team with complexity=2
        # Expected: runs to completion without raising BudgetExceededError
        # because specialist tokens are in task DB, not session DB
        result = await ts.run_team_v3(
            "Review transactions", domain="finance-audit", complexity=2
        )

        # If budget enforcement somehow caught the task-token usage, this would be unexpected.
        # The gap is that it should NOT catch it.
        assert (
            result.truncated is False
        ), (
            "UNEXPECTED: Budget caught specialist tokens despite task DB isolation. "
            "Gap may have been closed."
        )
        assert len(result.specialist_results) > 0, "Specialists should have run"


# ── Test E: Verify sequential returns (results, truncated) tuple correctly ─────


@pytest.mark.asyncio
async def test_sequential_run_specialists_returns_tuple() -> None:
    """Verify that _run_sequential returns (results, truncated) tuple correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = ClaudiumConfig(root=tmp_path, token_budget=100_000)

        response = "Transaction TXN-X $10k. SOX C-1. Risk flagged. Ref: INV-X."
        agent = ClaudiumAgent(
            config=config, harness=MockHarness(responses=[response, response])
        )

        from claudium.teams.session import TeamSession
        from claudium.teams.specialist import TRANSACTION_AUDITOR, RISK_ANALYST

        ts = TeamSession(agent=agent, session_id="seq-tuple")
        await ts._ensure_store()

        # Call _run_sequential directly with 2 specialists
        results, truncated = await ts._run_sequential(
            "Audit invoice", [TRANSACTION_AUDITOR, RISK_ANALYST], "finance-audit"
        )

        assert truncated is False, "Should not be truncated when within budget"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert all(
            hasattr(r, "specialist") and hasattr(r, "output") and hasattr(r, "fitness_score")
            for r in results
        ), "All results should be SpecialistResult objects"


@pytest.mark.asyncio
async def test_sequential_check_budget_between_each_specialist() -> None:
    """Verify that _check_budget() is called BEFORE each specialist in sequence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Budget of 500 tokens, no grace
        config = ClaudiumConfig(
            root=tmp_path, token_budget=500, budget_grace_pct=0.0
        )

        agent = ClaudiumAgent(
            config=config,
            harness=MockHarness(
                responses=["spec1", "spec2"],
                token_writer_session_db=tmp_path / "seq-check-between.db",
                tokens_per_call=(250, 250),  # Each call writes 500 tokens = budget
            ),
        )

        from claudium.teams.session import TeamSession
        from claudium.teams.specialist import TRANSACTION_AUDITOR, RISK_ANALYST

        ts = TeamSession(agent=agent, session_id="seq-check-between")
        ts.db_path = tmp_path / "seq-check-between.db"
        await ts._ensure_store()

        # Run _run_sequential with 2 specialists
        results, truncated = await ts._run_sequential(
            "Audit task", [TRANSACTION_AUDITOR, RISK_ANALYST], "finance-audit"
        )

        # First specialist should run fine (0 tokens initially)
        # After 1st specialist, MockHarness writes 500 tokens to session DB
        # 2nd specialist's pre-check should raise BudgetExceededError
        assert (
            truncated is True
        ), "Should be truncated when budget exceeded between specialists"
        assert (
            len(results) == 1
        ), f"Should have exactly 1 result (2nd specialist didn't run), got {len(results)}"


# ── Test F: Verify truncation with grace percentage ───────────────────────────


@pytest.mark.asyncio
async def test_sequential_budget_check_respects_grace_percentage() -> None:
    """Verify that budget checks use grace percentage correctly in sequential mode.

    This test verifies that the grace percentage is applied correctly:
    - Budget: 1000 tokens
    - Grace: 10% → grace_limit = 1000 * 1.10 = 1100 tokens
    - We seed 1000 tokens initially
    - Second specialist's pre-check sees 1000 tokens >= 1100? No, 1000 < 1100, so continues
    - But then we seed more tokens to exceed the grace limit
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Budget of 1000 tokens with 10% grace
        config = ClaudiumConfig(
            root=tmp_path, token_budget=1000, budget_grace_pct=0.10
        )

        agent = ClaudiumAgent(config=config, harness=MockHarness(responses=["spec1", "spec2"]))

        from claudium.teams.session import TeamSession
        from claudium.teams.specialist import TRANSACTION_AUDITOR, RISK_ANALYST

        ts = TeamSession(agent=agent, session_id="seq-grace")
        await ts._ensure_store()

        # Seed tokens to exceed grace limit (1100) but not yet at second check
        # Start with 1050 tokens (within grace limit)
        await _seed_tokens(ts.db_path, 525, 525)

        results, truncated = await ts._run_sequential(
            "Audit", [TRANSACTION_AUDITOR, RISK_ANALYST], "finance-audit"
        )

        # First specialist check: 1050 < 1100 (grace limit), so OK
        # After first specialist runs and completes, we still have 1050 tokens
        # (MockHarness doesn't write tokens in this scenario)
        # Second specialist check: 1050 < 1100, so OK → runs to completion
        assert truncated is False, "Should not be truncated when within grace limit"
        assert len(results) == 2, "Both specialists should have run"

        # Now test the opposite: seed tokens beyond grace limit
        ts2 = TeamSession(agent=agent, session_id="seq-grace-exceed")
        await ts2._ensure_store()

        # Seed tokens exceeding grace limit (> 1100)
        await _seed_tokens(ts2.db_path, 600, 520)  # 1120 tokens > 1100 grace limit

        results2, truncated2 = await ts2._run_sequential(
            "Audit", [TRANSACTION_AUDITOR, RISK_ANALYST], "finance-audit"
        )

        # First check: 1120 >= 1100 (grace limit), so truncate immediately
        assert truncated2 is True, "Should be truncated when grace limit exceeded at start"
        assert (
            len(results2) == 0
        ), "No specialists should have run when grace limit exceeded upfront"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
