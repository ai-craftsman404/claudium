"""Integration workflow tests for token budget enforcement with audit export and config cascade."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.audit import export_audit
from claudium.config import load_config
from claudium.core import ClaudiumAgent
from claudium.types import BudgetExceededError, ClaudiumConfig, ClaudiumEvent, HarnessResult

# ── Shared MockHarness ────────────────────────────────────────────────────────


class MockHarness:
    """Mock harness for testing without real API calls."""

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


# ── Helper functions ──────────────────────────────────────────────────────────


async def _seed_tokens(
    db_path: Path,
    session_id: str,
    input_tokens: int,
    output_tokens: int,
    skill: str = "test",
    model: str = "claude-test",
    created_at: str = "2026-05-10T10:00:00+00:00",
) -> None:
    """Insert a call_log row with the given token counts."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        await db.execute(
            "insert into call_log "
            "(session_id, skill, model, latency_ms, "
            "input_tokens, output_tokens, success, created_at) "
            "values (?,?,?,?,?,?,?,?)",
            (session_id, skill, model, 100.0, input_tokens, output_tokens, 1, created_at),
        )
        await db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Test A: Audit after full run
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_audit_after_full_run_json_format(tmp_path: Path) -> None:
    """Test A: Audit captures tokens from a full session run in JSON format."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s1")

    # Seed the session DB with known token counts
    await _seed_tokens(session.db_path, "s1", input_tokens=300, output_tokens=200)

    # Export audit
    result = await export_audit([session.db_path], fmt="json")
    data = json.loads(result)

    # Assert budget_consumed equals the seeded token sum
    assert data["budget_consumed"] == 500
    # Assert budget_limit is None (no explicit limit passed)
    assert data["budget_limit"] is None
    # Verify call_log entry
    assert len(data["call_log"]) == 1
    assert data["call_log"][0]["session_id"] == "s1"
    assert data["call_log"][0]["input_tokens"] == 300
    assert data["call_log"][0]["output_tokens"] == 200


@pytest.mark.asyncio
async def test_audit_multiple_calls_aggregated(tmp_path: Path) -> None:
    """Test A extended: Multiple calls in one session aggregate correctly."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s1-multi")

    # Seed multiple call_log entries
    async with aiosqlite.connect(session.db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        _cols = (
            "(session_id, skill, model, latency_ms,"
            " input_tokens, output_tokens, success, created_at)"
        )
        await db.execute(
            f"insert into call_log {_cols} values (?,?,?,?,?,?,?,?)",
            ("s1-multi", "skill1", "m", 100.0, 100, 50, 1, "2026-05-10T10:00:00+00:00"),
        )
        await db.execute(
            f"insert into call_log {_cols} values (?,?,?,?,?,?,?,?)",
            ("s1-multi", "skill2", "m", 150.0, 200, 100, 1, "2026-05-10T10:05:00+00:00"),
        )
        await db.commit()

    result = await export_audit([session.db_path], fmt="json")
    data = json.loads(result)

    assert data["budget_consumed"] == 450  # (100+50) + (200+100)
    assert len(data["call_log"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Test B: Audit with budget_limit param
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_audit_with_explicit_budget_limit(tmp_path: Path) -> None:
    """Test B: export_audit() accepts budget_limit and includes it in report."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s2")

    await _seed_tokens(session.db_path, "s2", input_tokens=300, output_tokens=200)

    result = await export_audit([session.db_path], budget_limit=50000, fmt="json")
    data = json.loads(result)

    assert data["budget_limit"] == 50000
    assert data["budget_consumed"] == 500
    assert data["budget_consumed"] < data["budget_limit"]


@pytest.mark.asyncio
async def test_audit_budget_limit_zero(tmp_path: Path) -> None:
    """Test B extended: budget_limit can be 0 (hard stop)."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s2-zero")

    await _seed_tokens(session.db_path, "s2-zero", input_tokens=100, output_tokens=50)

    result = await export_audit([session.db_path], budget_limit=0, fmt="json")
    data = json.loads(result)

    assert data["budget_limit"] == 0
    assert data["budget_consumed"] == 150


@pytest.mark.asyncio
async def test_audit_budget_limit_none_explicit(tmp_path: Path) -> None:
    """Test B extended: budget_limit=None keeps report as None."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s2-none")

    await _seed_tokens(session.db_path, "s2-none", input_tokens=100, output_tokens=50)

    result = await export_audit([session.db_path], budget_limit=None, fmt="json")
    data = json.loads(result)

    assert data["budget_limit"] is None
    assert data["budget_consumed"] == 150


# ══════════════════════════════════════════════════════════════════════════════
# Test C: Audit CSV includes budget fields
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_audit_csv_does_not_expose_budget_fields(tmp_path: Path) -> None:
    """Test C: CSV format omits budget_consumed/budget_limit (documenting current behavior)."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s3")

    await _seed_tokens(session.db_path, "s3", input_tokens=300, output_tokens=200)

    result = await export_audit([session.db_path], budget_limit=50000, fmt="csv")

    # CSV should have call_log section but no separate budget fields
    assert "# CALL LOG" in result
    assert "session_id" in result
    assert "input_tokens" in result
    assert "output_tokens" in result
    # Budget metadata is NOT in CSV rows
    assert "budget_consumed" not in result
    assert "budget_limit" not in result


@pytest.mark.asyncio
async def test_audit_csv_call_log_values_present(tmp_path: Path) -> None:
    """Test C extended: CSV includes token values in call_log section."""
    config = ClaudiumConfig(root=tmp_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())
    session = await agent.session("s3-csv")

    await _seed_tokens(session.db_path, "s3-csv", input_tokens=300, output_tokens=200)

    result = await export_audit([session.db_path], fmt="csv")

    # Token values should appear in the CSV rows
    lines = result.split("\n")
    data_lines = [line for line in lines if "s3-csv" in line]
    assert len(data_lines) > 0
    # Check that the token values appear
    assert "300" in result
    assert "200" in result


# ══════════════════════════════════════════════════════════════════════════════
# Test D: Three-layer budget cascade
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_budget_cascade_config_to_session(tmp_path: Path) -> None:
    """Test D: Session inherits budget from config when no explicit override."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session1 = await agent.session("s1")
    # Should inherit from config
    assert session1._token_budget == 100


@pytest.mark.asyncio
async def test_budget_cascade_explicit_override(tmp_path: Path) -> None:
    """Test D: session() param overrides config budget."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session2 = await agent.session("s2", token_budget=5000)
    # Should use explicit param
    assert session2._token_budget == 5000


@pytest.mark.asyncio
async def test_budget_cascade_fallback_to_config(tmp_path: Path) -> None:
    """Test D: session(token_budget=None) falls back to config budget."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session3 = await agent.session("s3", token_budget=None)
    # Should fall back to config
    assert session3._token_budget == 100


@pytest.mark.asyncio
async def test_budget_cascade_check_budget_above_threshold(tmp_path: Path) -> None:
    """Test D: _check_budget() raises when consuming above configured threshold."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("s-above")
    await _seed_tokens(session.db_path, "s-above", input_tokens=60, output_tokens=50)

    # 110 > 100 → should raise
    with pytest.raises(BudgetExceededError) as exc_info:
        await session._check_budget()
    assert exc_info.value.limit == 100


@pytest.mark.asyncio
async def test_budget_cascade_check_budget_below_threshold(tmp_path: Path) -> None:
    """Test D: _check_budget() allows when consuming below configured threshold."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("s-below")
    await _seed_tokens(session.db_path, "s-below", input_tokens=40, output_tokens=30)

    # 70 < 100 → should not raise
    await session._check_budget()


@pytest.mark.asyncio
async def test_budget_cascade_with_grace_pct(tmp_path: Path) -> None:
    """Test D: grace_pct from config allows overage."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100, budget_grace_pct=0.10)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("s-grace")
    # 105 < 110 (100 * 1.10) → within grace
    await _seed_tokens(session.db_path, "s-grace", input_tokens=60, output_tokens=45)
    await session._check_budget()  # should not raise


@pytest.mark.asyncio
async def test_budget_cascade_beyond_grace_pct(tmp_path: Path) -> None:
    """Test D: beyond grace_pct threshold raises."""
    config = ClaudiumConfig(root=tmp_path, token_budget=100, budget_grace_pct=0.10)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("s-beyond-grace")
    # 115 > 110 (100 * 1.10) → beyond grace
    await _seed_tokens(session.db_path, "s-beyond-grace", input_tokens=70, output_tokens=45)

    with pytest.raises(BudgetExceededError):
        await session._check_budget()


# ══════════════════════════════════════════════════════════════════════════════
# Test E: TOML config loading cascade
# ══════════════════════════════════════════════════════════════════════════════


def test_config_loading_from_toml_budget_section(tmp_path: Path) -> None:
    """Test E: load_config() reads [budget] section from claudium.toml."""
    toml_path = tmp_path / "claudium.toml"
    toml_path.write_text(
        "[budget]\ntoken_budget = 75000\nbudget_grace_pct = 0.05\n",
        encoding="utf-8",
    )

    config = load_config(toml_path)
    assert config.token_budget == 75000
    assert config.budget_grace_pct == pytest.approx(0.05)


def test_config_loading_budget_defaults(tmp_path: Path) -> None:
    """Test E: budget defaults when [budget] section absent."""
    toml_path = tmp_path / "claudium.toml"
    toml_path.write_text("[agent]\nmodel = 'claude-opus-4-5'\n", encoding="utf-8")

    config = load_config(toml_path)
    assert config.token_budget is None
    assert config.budget_grace_pct == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_config_cascade_agent_from_toml(tmp_path: Path) -> None:
    """Test E: Agent inherits budget from loaded TOML config."""
    toml_path = tmp_path / "claudium.toml"
    toml_path.write_text(
        "[budget]\ntoken_budget = 75000\nbudget_grace_pct = 0.05\n",
        encoding="utf-8",
    )

    config = load_config(toml_path)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("e-toml")
    assert session._token_budget == 75000
    assert session._budget_grace_pct == pytest.approx(0.05)


def test_config_partial_budget_section(tmp_path: Path) -> None:
    """Test E: partial [budget] section uses defaults for missing keys."""
    toml_path = tmp_path / "claudium.toml"
    # Only token_budget, no budget_grace_pct
    toml_path.write_text(
        "[budget]\ntoken_budget = 50000\n",
        encoding="utf-8",
    )

    config = load_config(toml_path)
    assert config.token_budget == 50000
    assert config.budget_grace_pct == pytest.approx(0.10)  # default


def test_config_empty_budget_section(tmp_path: Path) -> None:
    """Test E: empty [budget] section uses all defaults."""
    toml_path = tmp_path / "claudium.toml"
    toml_path.write_text(
        "[budget]\n",
        encoding="utf-8",
    )

    config = load_config(toml_path)
    assert config.token_budget is None
    assert config.budget_grace_pct == pytest.approx(0.10)


# ══════════════════════════════════════════════════════════════════════════════
# Test F: Multi-DB audit aggregation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_audit_multi_db_aggregates_tokens(tmp_path: Path) -> None:
    """Test F: export_audit([db1, db2]) aggregates budget_consumed from both DBs."""
    db1 = tmp_path / "session1.db"
    db2 = tmp_path / "session2.db"

    # Seed tokens in db1
    await _seed_tokens(db1, "s1", input_tokens=100, output_tokens=50)
    # Seed tokens in db2
    await _seed_tokens(db2, "s2", input_tokens=200, output_tokens=100)

    result = await export_audit([db1, db2], fmt="json")
    data = json.loads(result)

    # Should sum all tokens: (100+50) + (200+100) = 450
    assert data["budget_consumed"] == 450
    assert len(data["call_log"]) == 2


@pytest.mark.asyncio
async def test_audit_multi_db_with_budget_limit(tmp_path: Path) -> None:
    """Test F: budget_limit applies to aggregated consumption across DBs."""
    db1 = tmp_path / "m1.db"
    db2 = tmp_path / "m2.db"

    await _seed_tokens(db1, "s1", input_tokens=100, output_tokens=50)
    await _seed_tokens(db2, "s2", input_tokens=200, output_tokens=100)

    result = await export_audit([db1, db2], budget_limit=1000, fmt="json")
    data = json.loads(result)

    assert data["budget_consumed"] == 450
    assert data["budget_limit"] == 1000


@pytest.mark.asyncio
async def test_audit_multi_db_different_sessions(tmp_path: Path) -> None:
    """Test F: multiple DBs can represent different session IDs."""
    db1 = tmp_path / "sess-a.db"
    db2 = tmp_path / "sess-b.db"

    await _seed_tokens(db1, "sess-a", input_tokens=100, output_tokens=50)
    await _seed_tokens(db2, "sess-b", input_tokens=300, output_tokens=150)

    result = await export_audit([db1, db2], fmt="json")
    data = json.loads(result)

    assert len(data["call_log"]) == 2
    assert {e["session_id"] for e in data["call_log"]} == {"sess-a", "sess-b"}
    assert data["budget_consumed"] == 600


@pytest.mark.asyncio
async def test_audit_multi_db_one_missing(tmp_path: Path) -> None:
    """Test F: missing DB paths are skipped gracefully."""
    db1 = tmp_path / "exists.db"
    db2 = tmp_path / "ghost.db"

    await _seed_tokens(db1, "s1", input_tokens=100, output_tokens=50)
    # db2 doesn't exist

    result = await export_audit([db1, db2], fmt="json")
    data = json.loads(result)

    assert data["budget_consumed"] == 150
    assert len(data["call_log"]) == 1


@pytest.mark.asyncio
async def test_audit_multi_db_all_missing(tmp_path: Path) -> None:
    """Test F: all missing DBs returns empty but valid report."""
    db1 = tmp_path / "ghost1.db"
    db2 = tmp_path / "ghost2.db"

    result = await export_audit([db1, db2], fmt="json")
    data = json.loads(result)

    assert data["budget_consumed"] == 0
    assert data["call_log"] == []
    assert "generated_at" in data


@pytest.mark.asyncio
async def test_audit_multi_db_null_tokens(tmp_path: Path) -> None:
    """Test F: NULL token values are treated as 0 in aggregation."""
    db1 = tmp_path / "null1.db"

    async with aiosqlite.connect(db1) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        _cols = (
            "(session_id, skill, model, latency_ms,"
            " input_tokens, output_tokens, success, created_at)"
        )
        await db.execute(
            f"insert into call_log {_cols} values (?,?,?,?,?,?,?,?)",
            ("s1", "test", "m", 100.0, None, None, 1, "2026-05-10T10:00:00+00:00"),
        )
        await db.commit()

    result = await export_audit([db1], fmt="json")
    data = json.loads(result)

    # NULL + NULL = 0
    assert data["budget_consumed"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Full workflow with audit export
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_full_workflow_config_to_audit(tmp_path: Path) -> None:
    """Integration: Load TOML config → create agent → session → seed tokens → audit export."""
    # Step 1: Create TOML config
    toml_path = tmp_path / "claudium.toml"
    toml_path.write_text(
        "[budget]\ntoken_budget = 100000\nbudget_grace_pct = 0.05\n",
        encoding="utf-8",
    )

    # Step 2: Load config
    config = load_config(toml_path)
    assert config.token_budget == 100000

    # Step 3: Create agent
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    # Step 4: Create session
    session = await agent.session("workflow-s1")
    assert session._token_budget == 100000

    # Step 5: Seed tokens (simulating call_log entries)
    await _seed_tokens(session.db_path, "workflow-s1", input_tokens=500, output_tokens=300)

    # Step 6: Export audit
    result = await export_audit([session.db_path], fmt="json")
    data = json.loads(result)

    assert data["budget_consumed"] == 800
    assert data["budget_limit"] is None


@pytest.mark.asyncio
async def test_full_workflow_budget_enforcement(tmp_path: Path) -> None:
    """Integration: Config budget → session → verify enforcement."""
    # Create config with tight budget
    config = ClaudiumConfig(root=tmp_path, token_budget=200, budget_grace_pct=0.0)
    agent = ClaudiumAgent(config=config, harness=MockHarness())

    session = await agent.session("enforce-s1")

    # Seed tokens below limit
    await _seed_tokens(session.db_path, "enforce-s1", input_tokens=80, output_tokens=50)
    await session._check_budget()  # Should pass (130 < 200)

    # Seed more tokens to exceed limit
    async with aiosqlite.connect(session.db_path) as db:
        _cols = (
            "(session_id, skill, model, latency_ms,"
            " input_tokens, output_tokens, success, created_at)"
        )
        await db.execute(
            f"insert into call_log {_cols} values (?,?,?,?,?,?,?,?)",
            ("enforce-s1", "test2", "m", 100.0, 80, 50, 1, "2026-05-10T10:05:00+00:00"),
        )
        await db.commit()

    # Now should fail (260 > 200)
    with pytest.raises(BudgetExceededError):
        await session._check_budget()

    # But audit export still works and reports consumption
    result = await export_audit([session.db_path], fmt="json")
    data = json.loads(result)
    assert data["budget_consumed"] == 260
