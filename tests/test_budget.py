"""TDD tests for token budget enforcement — written before implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.types import BudgetExceededError, ClaudiumConfig, ClaudiumEvent, HarnessResult

# ── Shared helpers ────────────────────────────────────────────────────────────


class MockHarness:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def run(self, *, prompt, system_prompt, config, **_) -> HarnessResult:
        if self._responses and self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = "mock response"
        self._call_count += 1
        return HarnessResult(text=text)

    async def stream(self, **_) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "mock"})


async def _seed_tokens(db_path: Path, input_tokens: int, output_tokens: int) -> None:
    """Insert a call_log row with the given token counts."""
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
            ("test-session", "test", "m", 100.0,
             input_tokens, output_tokens, 1, "2026-05-10T10:00:00+00:00"),
        )
        await db.commit()


# ── BudgetExceededError ───────────────────────────────────────────────────────


def test_budget_exceeded_error_message() -> None:
    err = BudgetExceededError(consumed=1100, limit=1000, session_id="sess-1")
    assert "1100" in str(err)
    assert "1000" in str(err)
    assert "sess-1" in str(err)
    assert err.consumed == 1100
    assert err.limit == 1000
    assert err.session_id == "sess-1"


# ── _check_budget() unit tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_none_never_raises(tmp_path: Path) -> None:
    config = ClaudiumConfig(root=tmp_path)  # no token_budget
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-none")
    await _seed_tokens(session.db_path, 1_000_000, 1_000_000)
    await session._check_budget()  # must not raise


@pytest.mark.asyncio
async def test_budget_under_limit_allows(tmp_path: Path) -> None:
    config = ClaudiumConfig(root=tmp_path, token_budget=1000)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-under")
    await _seed_tokens(session.db_path, 400, 400)  # 800 < 1000
    await session._check_budget()  # must not raise


@pytest.mark.asyncio
async def test_budget_exceeded_raises(tmp_path: Path) -> None:
    config = ClaudiumConfig(root=tmp_path, token_budget=1000, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-over")
    await _seed_tokens(session.db_path, 600, 500)  # 1100 > 1000
    with pytest.raises(BudgetExceededError) as exc_info:
        await session._check_budget()
    assert exc_info.value.consumed == 1100
    assert exc_info.value.limit == 1000


@pytest.mark.asyncio
async def test_budget_within_grace_allows(tmp_path: Path) -> None:
    config = ClaudiumConfig(root=tmp_path, token_budget=1000, budget_grace_pct=0.10)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-grace")
    await _seed_tokens(session.db_path, 600, 450)  # 1050 < 1100 grace limit
    await session._check_budget()  # within grace — must not raise


@pytest.mark.asyncio
async def test_budget_beyond_grace_raises(tmp_path: Path) -> None:
    config = ClaudiumConfig(root=tmp_path, token_budget=1000, budget_grace_pct=0.10)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-beyond-grace")
    await _seed_tokens(session.db_path, 700, 500)  # 1200 > 1100 grace limit
    with pytest.raises(BudgetExceededError):
        await session._check_budget()


@pytest.mark.asyncio
async def test_budget_counts_failed_calls(tmp_path: Path) -> None:
    """Failed calls (success=0) must still count toward budget."""
    config = ClaudiumConfig(root=tmp_path, token_budget=500, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-failed")
    async with aiosqlite.connect(session.db_path) as db:
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
            ("sess-failed", "test", "m", 100.0, 300, 250, 0, "2026-05-10T10:00:00+00:00"),
        )
        await db.commit()
    with pytest.raises(BudgetExceededError):
        await session._check_budget()


# ── session() param override ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_param_overrides_config_budget(tmp_path: Path) -> None:
    """session(token_budget=X) takes precedence over ClaudiumConfig.token_budget."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-override", token_budget=2000)
    await _seed_tokens(session.db_path, 800, 700)  # 1500 > 100 but < 2000
    await session._check_budget()  # must not raise (2000 budget applies)


@pytest.mark.asyncio
async def test_session_param_none_falls_back_to_config(tmp_path: Path) -> None:
    """session(token_budget=None) falls back to ClaudiumConfig.token_budget."""
    config = ClaudiumConfig(root=tmp_path, token_budget=500, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("sess-fallback")  # no override → uses config budget
    await _seed_tokens(session.db_path, 300, 250)  # 550 > 500
    with pytest.raises(BudgetExceededError):
        await session._check_budget()


# ── TeamRunV3Result truncated flag ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_team_run_truncated_on_budget(tmp_path: Path) -> None:
    """run_team_v3 sets truncated=True when budget exceeded mid-run."""
    from claudium.teams.session import TeamSession

    config = ClaudiumConfig(root=tmp_path, token_budget=1, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    ts = TeamSession(agent=agent, session_id="ts-budget-1")

    # Seed session db with tokens at limit so first specialist check fails
    await ts._ensure_store()
    await _seed_tokens(ts.db_path, 1, 0)  # 1 token consumed = at limit

    result = await ts.run_team_v3("Review invoice", domain="finance-audit", complexity=2)
    assert result.truncated is True
    assert len(result.specialist_results) < 2


@pytest.mark.asyncio
async def test_team_run_not_truncated_within_budget(tmp_path: Path) -> None:
    """run_team_v3 sets truncated=False when within budget."""
    from claudium.teams.session import TeamSession

    good_text = (
        "Transaction TXN-007 $25k. SOX control C-3. "
        "High risk anomaly flagged. Per invoice #INV-007."
    )
    config = ClaudiumConfig(root=tmp_path, token_budget=100_000)
    agent = ClaudiumAgent(config=config, harness=MockHarness([good_text]))
    ts = TeamSession(agent=agent, session_id="ts-budget-ok")
    result = await ts.run_team_v3("Review invoice", domain="finance-audit", complexity=1)
    assert result.truncated is False


# ── Config from claudium.toml ─────────────────────────────────────────────────


def test_budget_loaded_from_toml(tmp_path: Path) -> None:
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n\n'
        "[budget]\ntoken_budget = 50000\nbudget_grace_pct = 0.05\n",
        encoding="utf-8",
    )
    from claudium.config import load_config
    config = load_config(tmp_path / "claudium.toml")
    assert config.token_budget == 50000
    assert config.budget_grace_pct == pytest.approx(0.05)


def test_budget_defaults_when_not_in_toml(tmp_path: Path) -> None:
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8"
    )
    from claudium.config import load_config
    config = load_config(tmp_path / "claudium.toml")
    assert config.token_budget is None
    assert config.budget_grace_pct == pytest.approx(0.10)


# ── Audit export includes budget fields ───────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_report_includes_budget_fields(tmp_path: Path) -> None:
    import json

    from claudium.audit import export_audit
    db = tmp_path / "budget-audit.db"
    await _seed_tokens(db, 300, 200)
    result = await export_audit([db], fmt="json")
    data = json.loads(result)
    assert "budget_consumed" in data
    assert data["budget_consumed"] == 500
    assert "budget_limit" in data


@pytest.mark.asyncio
async def test_audit_budget_limit_reflects_config(tmp_path: Path) -> None:
    import json

    from claudium.audit import export_audit
    db = tmp_path / "budget-limit.db"
    await _seed_tokens(db, 100, 100)
    result = await export_audit([db], budget_limit=1000, fmt="json")
    data = json.loads(result)
    assert data["budget_limit"] == 1000
