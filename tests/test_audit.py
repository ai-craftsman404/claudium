"""Tests for claudium.audit — compliance audit log export (unit + e2e CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest
from typer.testing import CliRunner

from claudium.audit import export_audit
from claudium.cli import app

runner = CliRunner()


async def _seed_call_log(db_path: Path, rows: list[tuple]) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
        cols = "(session_id, skill, model, latency_ms, input_tokens," \
               " output_tokens, success, created_at)"
        for row in rows:
            await db.execute(
                f"insert into call_log {cols} values (?,?,?,?,?,?,?,?)", row
            )
        await db.commit()


async def _seed_team_run(db_path: Path, run_id: str, domain: str, created_at: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists team_runs_v3 ("
            "id text primary key, prompt text, domain text, specialists_used text, "
            "adjudication text, synthesis text, created_at text)"
        )
        await db.execute(
            "create table if not exists specialist_runs ("
            "id integer primary key autoincrement, run_id text, "
            "specialist_name text, output_text text, fitness_score real)"
        )
        tr_cols = "(id, prompt, domain, specialists_used, adjudication, synthesis, created_at)"
        await db.execute(
            f"insert into team_runs_v3 {tr_cols} values (?,?,?,?,?,?,?)",
            (
                run_id, "Review invoice", domain,
                '["transaction-auditor"]',
                '{"mode": "rule-based", "accepted": true, "gaps": [], "contradictions": []}',
                None, created_at,
            ),
        )
        await db.execute(
            "insert into specialist_runs(run_id, specialist_name, output_text, fitness_score)"
            " values (?,?,?,?)",
            (run_id, "transaction-auditor", "Transaction TXN-1. SOX. High risk.", 0.9),
        )
        await db.commit()


# ── JSON format ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_json_empty_db(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    result = await export_audit([db], fmt="json")
    data = json.loads(result)
    assert data["call_log"] == []
    assert data["team_runs"] == []
    assert "generated_at" in data


@pytest.mark.asyncio
async def test_export_json_call_log_entries(tmp_path: Path) -> None:
    db = tmp_path / "session.db"
    await _seed_call_log(db, [
        ("sess-1", "triage", "claude-sonnet-4-5", 123.4, 100, 50, 1, "2026-05-09T10:00:00+00:00"),
        ("sess-1", "review", "claude-sonnet-4-5", 200.0, 200, 80, 1, "2026-05-09T10:01:00+00:00"),
    ])
    result = await export_audit([db], fmt="json")
    data = json.loads(result)
    assert len(data["call_log"]) == 2
    assert data["call_log"][0]["session_id"] == "sess-1"
    assert data["call_log"][0]["skill"] == "triage"
    assert data["call_log"][0]["success"] == 1


@pytest.mark.asyncio
async def test_export_json_team_runs_with_specialists(tmp_path: Path) -> None:
    db = tmp_path / "teams.db"
    await _seed_team_run(db, "run-001", "finance-audit", "2026-05-09T10:00:00+00:00")
    result = await export_audit([db], fmt="json")
    data = json.loads(result)
    assert len(data["team_runs"]) == 1
    run = data["team_runs"][0]
    assert run["run_id"] == "run-001"
    assert run["domain"] == "finance-audit"
    assert run["adjudication"]["mode"] == "rule-based"
    assert len(run["specialist_runs"]) == 1
    assert run["specialist_runs"][0]["specialist_name"] == "transaction-auditor"
    assert run["specialist_runs"][0]["fitness_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_export_generated_at_is_utc_iso(tmp_path: Path) -> None:
    from datetime import datetime
    db = tmp_path / "ts.db"
    result = await export_audit([db], fmt="json")
    data = json.loads(result)
    # Must parse without error
    dt = datetime.fromisoformat(data["generated_at"])
    assert dt.tzinfo is not None


# ── Filters ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_session_filter(tmp_path: Path) -> None:
    db = tmp_path / "multi.db"
    await _seed_call_log(db, [
        ("sess-A", "triage", "claude-sonnet-4-5", 100.0, 50, 20, 1, "2026-05-09T10:00:00+00:00"),
        ("sess-B", "review", "claude-sonnet-4-5", 150.0, 80, 30, 1, "2026-05-09T10:01:00+00:00"),
    ])
    result = await export_audit([db], session="sess-A", fmt="json")
    data = json.loads(result)
    assert len(data["call_log"]) == 1
    assert data["call_log"][0]["session_id"] == "sess-A"
    assert data["filters"]["session"] == "sess-A"


@pytest.mark.asyncio
async def test_export_since_filter(tmp_path: Path) -> None:
    db = tmp_path / "dated.db"
    await _seed_call_log(db, [
        ("s1", "old-skill", "claude-sonnet-4-5", 100.0, 10, 5, 1, "2026-04-01T00:00:00+00:00"),
        ("s1", "new-skill", "claude-sonnet-4-5", 200.0, 20, 10, 1, "2026-05-09T00:00:00+00:00"),
    ])
    result = await export_audit([db], since="2026-05-01", fmt="json")
    data = json.loads(result)
    assert len(data["call_log"]) == 1
    assert data["call_log"][0]["skill"] == "new-skill"


# ── CSV format ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_csv_call_log_headers(tmp_path: Path) -> None:
    db = tmp_path / "csv.db"
    await _seed_call_log(db, [
        ("sess-1", "triage", "claude-sonnet-4-5", 100.0, 50, 20, 1, "2026-05-09T10:00:00+00:00"),
    ])
    result = await export_audit([db], fmt="csv")
    assert "session_id" in result
    assert "latency_ms" in result
    assert "input_tokens" in result
    assert "sess-1" in result


@pytest.mark.asyncio
async def test_export_csv_team_runs_section(tmp_path: Path) -> None:
    db = tmp_path / "csv_teams.db"
    await _seed_team_run(db, "run-csv-1", "finance-audit", "2026-05-09T10:00:00+00:00")
    result = await export_audit([db], fmt="csv")
    assert "# TEAM RUNS V3" in result
    assert "run-csv-1" in result
    assert "finance-audit" in result
    assert "rule-based" in result


# ── E2E CLI tests ─────────────────────────────────────────────────────────────


def _mock_anthropic() -> MagicMock:
    mock_cls = MagicMock()
    instance = MagicMock()
    mock_cls.return_value = instance
    return mock_cls


def _setup_project(tmp_path: Path) -> Path:
    """Write claudium.toml and return the seeded state_dir."""
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8"
    )
    state_dir = tmp_path / ".claudium" / "sessions"
    state_dir.mkdir(parents=True)
    return state_dir


async def _seed_state_dir(state_dir: Path) -> None:
    db = state_dir / "audit-session.db"
    await _seed_call_log(db, [
        ("audit-session", "triage", "claude-sonnet-4-5", 142.3, 120, 60, 1,
         "2026-05-09T10:00:00+00:00"),
        ("audit-session", "review", "claude-sonnet-4-5", 230.1, 200, 90, 1,
         "2026-05-09T10:05:00+00:00"),
    ])
    await _seed_team_run(db, "run-e2e-1", "finance-audit", "2026-05-09T10:10:00+00:00")


def test_audit_export_cli_json_stdout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_dir = _setup_project(tmp_path)
    import asyncio
    asyncio.run(_seed_state_dir(state_dir))
    with patch("anthropic.AsyncAnthropic", _mock_anthropic()):
        result = runner.invoke(app, ["audit", "export"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["call_log"]) == 2
    assert data["call_log"][0]["skill"] == "triage"
    assert len(data["team_runs"]) == 1
    assert data["team_runs"][0]["domain"] == "finance-audit"


def test_audit_export_cli_csv_stdout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_dir = _setup_project(tmp_path)
    import asyncio
    asyncio.run(_seed_state_dir(state_dir))
    with patch("anthropic.AsyncAnthropic", _mock_anthropic()):
        result = runner.invoke(app, ["audit", "export", "--format", "csv"])
    assert result.exit_code == 0
    assert "# CALL LOG" in result.output
    assert "# TEAM RUNS V3" in result.output
    assert "triage" in result.output
    assert "finance-audit" in result.output


def test_audit_export_cli_to_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_dir = _setup_project(tmp_path)
    import asyncio
    asyncio.run(_seed_state_dir(state_dir))
    out_file = tmp_path / "report.json"
    with patch("anthropic.AsyncAnthropic", _mock_anthropic()):
        result = runner.invoke(app, ["audit", "export", "--output", str(out_file)])
    assert result.exit_code == 0
    assert "written to" in result.output
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "call_log" in data


def test_audit_export_cli_session_filter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state_dir = _setup_project(tmp_path)
    import asyncio
    asyncio.run(_seed_state_dir(state_dir))
    with patch("anthropic.AsyncAnthropic", _mock_anthropic()):
        result = runner.invoke(app, ["audit", "export", "--session", "audit-session"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert all(e["session_id"] == "audit-session" for e in data["call_log"])


def test_audit_export_cli_invalid_format(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_project(tmp_path)
    with patch("anthropic.AsyncAnthropic", _mock_anthropic()):
        result = runner.invoke(app, ["audit", "export", "--format", "xml"])
    assert result.exit_code != 0
    assert "json" in result.output or "csv" in result.output
