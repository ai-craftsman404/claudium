"""TDD tests for Claudium v3d user identity attribution.

All tests are written BEFORE implementation — they define the contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from claudium.core import ClaudiumAgent
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult

# ── MockHarness ───────────────────────────────────────────────────────────────


class MockHarness:
    async def run(self, *, prompt: str, system_prompt: str, config: Any, **_) -> HarnessResult:
        return HarnessResult(text="mock response")

    async def stream(
        self, *, prompt: str, system_prompt: str, config: Any, **_
    ) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": "mock response"})


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path: Path) -> ClaudiumConfig:
    cfg = ClaudiumConfig(root=tmp_path)
    (tmp_path / ".claudium" / "sessions").mkdir(parents=True)
    return cfg


@pytest.fixture
def agent(config: ClaudiumConfig) -> ClaudiumAgent:
    return ClaudiumAgent(config=config, harness=MockHarness())


# ── Tests 1–3: ClaudiumAgent.session() user_id plumbing ──────────────────────


@pytest.mark.asyncio
async def test_session_accepts_user_id(agent: ClaudiumAgent) -> None:
    """ClaudiumAgent.session() must accept a user_id keyword argument without raising."""
    session = await agent.session("s1", user_id="alice")
    assert session is not None


@pytest.mark.asyncio
async def test_session_user_id_stored(agent: ClaudiumAgent) -> None:
    """After session('s1', user_id='alice'), session._user_id must equal 'alice'."""
    session = await agent.session("s1", user_id="alice")
    assert session._user_id == "alice"


@pytest.mark.asyncio
async def test_session_user_id_defaults_none(agent: ClaudiumAgent) -> None:
    """session('s1') without user_id must default _user_id to None."""
    session = await agent.session("s1")
    assert session._user_id is None


# ── Tests 4–6: call_log DB column ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_log_user_id_column_exists(agent: ClaudiumAgent) -> None:
    """After session creation, call_log must contain a user_id column."""
    session = await agent.session("s1", user_id="alice")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("PRAGMA table_info(call_log)")
        columns = [row[1] for row in await cursor.fetchall()]
    assert "user_id" in columns


@pytest.mark.asyncio
async def test_call_log_records_user_id(agent: ClaudiumAgent) -> None:
    """After a prompt() run, the call_log row must contain user_id = 'alice'."""
    session = await agent.session("s1", user_id="alice")
    await session.prompt("hello")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT user_id FROM call_log ORDER BY id DESC LIMIT 1")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "alice"


@pytest.mark.asyncio
async def test_call_log_null_user_id_when_not_set(agent: ClaudiumAgent) -> None:
    """When session has no user_id, call_log.user_id must be NULL."""
    session = await agent.session("s1")
    await session.prompt("hello")
    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("SELECT user_id FROM call_log ORDER BY id DESC LIMIT 1")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] is None


# ── Test 7: migration safety ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_safe_existing_db(agent: ClaudiumAgent) -> None:
    """Creating a session on a DB missing user_id must add the column without error."""
    db_path = agent.state_dir / "s_migrate.db"
    # Seed an old-style call_log table without user_id
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        await db.commit()

    # Creating a new session must add the column and not raise
    session = await agent.session("s_migrate", user_id="bob")
    await session.prompt("hello")

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("PRAGMA table_info(call_log)")
        columns = [row[1] for row in await cursor.fetchall()]
        assert "user_id" in columns

        cursor = await db.execute("SELECT user_id FROM call_log ORDER BY id DESC LIMIT 1")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "bob"


# ── Tests 8–9: team_runs_v3 / specialist_runs columns ────────────────────────


@pytest.mark.asyncio
async def test_team_runs_v3_user_id_column(agent: ClaudiumAgent) -> None:
    """team_runs_v3 table must contain a user_id column after session creation."""
    session = await agent.session("s1", user_id="alice")
    async with aiosqlite.connect(session.db_path) as db:
        # Ensure team_runs_v3 table exists (session creation should handle this)
        await db.execute(
            "create table if not exists team_runs_v3 ("
            "id text primary key, prompt text, domain text, specialists_used text, "
            "adjudication text, synthesis text, created_at text)"
        )
        await db.commit()
        # Trigger migration via session's _ensure_store which should add user_id
        await session._ensure_store()
        cursor = await db.execute("PRAGMA table_info(team_runs_v3)")
        columns = [row[1] for row in await cursor.fetchall()]
    assert "user_id" in columns


@pytest.mark.asyncio
async def test_specialist_runs_user_id_column(agent: ClaudiumAgent) -> None:
    """specialist_runs table must contain a user_id column after session creation."""
    session = await agent.session("s1", user_id="alice")
    async with aiosqlite.connect(session.db_path) as db:
        # Ensure specialist_runs table exists
        await db.execute(
            "create table if not exists specialist_runs ("
            "id integer primary key autoincrement, run_id text, "
            "specialist_name text, output_text text, fitness_score real)"
        )
        await db.commit()
        # Trigger migration via _ensure_store
        await session._ensure_store()
        cursor = await db.execute("PRAGMA table_info(specialist_runs)")
        columns = [row[1] for row in await cursor.fetchall()]
    assert "user_id" in columns


# ── Tests 10–12: export_audit() user_id field and filter ─────────────────────


async def _seed_call_log_with_user(
    db_path: Path,
    rows: list[tuple[str, str | None]],  # (session_id, user_id)
) -> None:
    """Seed call_log rows that include a user_id column."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text, user_id text)"
        )
        for session_id, user_id in rows:
            await db.execute(
                "insert into call_log"
                "(session_id, skill, model, latency_ms, input_tokens,"
                " output_tokens, success, created_at, user_id)"
                " values (?,?,?,?,?,?,?,?,?)",
                (session_id, None, "claude-sonnet-4-5", 123.0, 10, 5, 1,
                 "2025-01-01T00:00:00+00:00", user_id),
            )
        await db.commit()


@pytest.mark.asyncio
async def test_audit_report_user_id_field(tmp_path: Path) -> None:
    """export_audit() JSON output must include a user_id field on call_log entries."""
    from claudium.audit import export_audit

    db = tmp_path / "audit_uid.db"
    await _seed_call_log_with_user(db, [("s1", "alice")])

    result = await export_audit([db])
    data = json.loads(result)
    assert len(data["call_log"]) == 1
    entry = data["call_log"][0]
    assert "user_id" in entry
    assert entry["user_id"] == "alice"


@pytest.mark.asyncio
async def test_audit_filter_by_user_id(tmp_path: Path) -> None:
    """export_audit(user_id='alice') must return only alice's call_log entries."""
    from claudium.audit import export_audit

    db = tmp_path / "audit_filter.db"
    await _seed_call_log_with_user(db, [("s1", "alice"), ("s2", "bob"), ("s3", "alice")])

    result = await export_audit([db], user_id="alice")
    data = json.loads(result)
    assert len(data["call_log"]) == 2
    for entry in data["call_log"]:
        assert entry["user_id"] == "alice"


@pytest.mark.asyncio
async def test_audit_filter_no_match(tmp_path: Path) -> None:
    """export_audit(user_id='nobody') must return an empty call_log list."""
    from claudium.audit import export_audit

    db = tmp_path / "audit_nomatch.db"
    await _seed_call_log_with_user(db, [("s1", "alice"), ("s2", "bob")])

    result = await export_audit([db], user_id="nobody")
    data = json.loads(result)
    assert data["call_log"] == []
