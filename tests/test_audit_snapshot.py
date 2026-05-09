"""Snapshot tests for claudium.audit — pin exact output format of export_audit."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest

from claudium.audit import export_audit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

# ---------------------------------------------------------------------------
# Seed helpers (deterministic, fixed values — no UUIDs, fixed timestamps)
# ---------------------------------------------------------------------------

FIXED_CREATED_AT = "2026-01-15T09:00:00+00:00"
FIXED_RUN_ID = "run-snap-001"


async def _create_db(db_path: Path) -> None:
    """Create all required tables in a fresh db."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists call_log ("
            "id integer primary key autoincrement, session_id text, skill text, "
            "model text, latency_ms real, input_tokens integer, output_tokens integer, "
            "success integer default 1, created_at text)"
        )
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
        await db.commit()


async def _seed_call_log_row(db_path: Path) -> None:
    """Insert one deterministic call_log row."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into call_log "
            "(session_id, skill, model, latency_ms, "
            "input_tokens, output_tokens, success, created_at) "
            "values (?,?,?,?,?,?,?,?)",
            (
                "snap-session",
                "triage",
                "claude-sonnet-4-5",
                123.4,
                100,
                50,
                1,
                FIXED_CREATED_AT,
            ),
        )
        await db.commit()


async def _seed_team_run_row(db_path: Path) -> None:
    """Insert one deterministic team_run + one specialist_run row."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into team_runs_v3 "
            "(id, prompt, domain, specialists_used, adjudication, synthesis, created_at) "
            "values (?,?,?,?,?,?,?)",
            (
                FIXED_RUN_ID,
                "Review invoice INV-001",
                "finance-audit",
                '["transaction-auditor","risk-assessor"]',
                '{"mode": "rule-based", "accepted": true, "gaps": [], "contradictions": []}',
                "Synthesis: all controls satisfied.",
                FIXED_CREATED_AT,
            ),
        )
        await db.execute(
            "insert into specialist_runs (run_id, specialist_name, output_text, fitness_score) "
            "values (?,?,?,?)",
            (FIXED_RUN_ID, "transaction-auditor", "SOX compliance verified.", 0.95),
        )
        await db.commit()


def _strip_generated_at_from_dict(data: dict) -> dict:
    """Remove generated_at from a parsed JSON dict (timestamp varies)."""
    data.pop("generated_at", None)
    return data


def _strip_generated_at_from_csv(csv_text: str) -> str:
    """Remove any line that contains generated_at from CSV (timestamp varies).
    The CSV format does not include a generated_at line, but guard anyway.
    Normalise line endings to LF so golden files are portable."""
    csv_text = csv_text.replace("\r\n", "\n")
    lines = [ln for ln in csv_text.splitlines(keepends=True) if "generated_at" not in ln]
    return "".join(lines)


def _load_or_create_snapshot(path: Path, content: str) -> str:
    """Return golden content; write it if the file does not yet exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — JSON call_log structure snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_json_call_log_structure(tmp_path: Path) -> None:
    """Seed one call_log row; export JSON; compare to golden file (stripped of generated_at)."""
    db = tmp_path / "snap_call_log.db"
    await _create_db(db)
    await _seed_call_log_row(db)

    raw = await export_audit([db], fmt="json")
    data = json.loads(raw)
    data = _strip_generated_at_from_dict(data)

    # Serialise deterministically for comparison
    actual = json.dumps(data, indent=2, sort_keys=True)

    golden_path = SNAPSHOTS_DIR / "audit_call_log.json"
    golden = _load_or_create_snapshot(golden_path, actual)

    assert actual == golden, (
        f"Snapshot mismatch for {golden_path}.\n"
        "If the format intentionally changed, delete the golden file and re-run to regenerate.\n"
        f"--- expected ---\n{golden}\n--- actual ---\n{actual}"
    )


# ---------------------------------------------------------------------------
# Test 2 — JSON team_run structure snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_json_team_run_structure(tmp_path: Path) -> None:
    """Seed one team_run + one specialist_run; export JSON; compare to golden file."""
    db = tmp_path / "snap_team_run.db"
    await _create_db(db)
    await _seed_team_run_row(db)

    raw = await export_audit([db], fmt="json")
    data = json.loads(raw)
    data = _strip_generated_at_from_dict(data)

    actual = json.dumps(data, indent=2, sort_keys=True)

    golden_path = SNAPSHOTS_DIR / "audit_team_run.json"
    golden = _load_or_create_snapshot(golden_path, actual)

    assert actual == golden, (
        f"Snapshot mismatch for {golden_path}.\n"
        "If the format intentionally changed, delete the golden file and re-run to regenerate.\n"
        f"--- expected ---\n{golden}\n--- actual ---\n{actual}"
    )


# ---------------------------------------------------------------------------
# Test 3 — CSV structure snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_csv_structure(tmp_path: Path) -> None:
    """Seed both call_log and team_run; export CSV; compare to golden file."""
    db = tmp_path / "snap_csv.db"
    await _create_db(db)
    await _seed_call_log_row(db)
    await _seed_team_run_row(db)

    actual_csv = await export_audit([db], fmt="csv")
    actual_csv = _strip_generated_at_from_csv(actual_csv)

    golden_path = SNAPSHOTS_DIR / "audit_report.csv"
    golden = _load_or_create_snapshot(golden_path, actual_csv)

    assert actual_csv == golden, (
        f"Snapshot mismatch for {golden_path}.\n"
        "If the format intentionally changed, delete the golden file and re-run to regenerate.\n"
        f"--- expected ---\n{golden}\n--- actual ---\n{actual_csv}"
    )
