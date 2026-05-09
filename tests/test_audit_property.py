"""Property-based tests for claudium.audit using Hypothesis."""

from __future__ import annotations

import asyncio
import itertools
import json
import tempfile
from pathlib import Path

import aiosqlite
from hypothesis import given, settings
from hypothesis import strategies as st

from claudium.audit import export_audit

# ---------------------------------------------------------------------------
# DDL constants
# ---------------------------------------------------------------------------

_CALL_LOG_DDL = (
    "create table if not exists call_log ("
    "id integer primary key autoincrement, session_id text, skill text, "
    "model text, latency_ms real, input_tokens integer, output_tokens integer, "
    "success integer default 1, created_at text)"
)

_TEAM_RUNS_DDL = (
    "create table if not exists team_runs_v3 ("
    "id text primary key, prompt text, domain text, specialists_used text, "
    "adjudication text, synthesis text, created_at text)"
)

_SPECIALIST_RUNS_DDL = (
    "create table if not exists specialist_runs ("
    "id integer primary key autoincrement, run_id text, "
    "specialist_name text, output_text text, fitness_score real)"
)

# ---------------------------------------------------------------------------
# Shared counter for unique db filenames across Hypothesis examples
# ---------------------------------------------------------------------------

_counter = itertools.count()


def _fresh_db() -> Path:
    """Return a unique temp-file path for a fresh SQLite db."""
    tmp_dir = Path(tempfile.gettempdir()) / "claudium_prop_tests"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"prop_{next(_counter)}.db"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Printable text using safe character categories, length 1-50
_printable_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" ",
    ),
    min_size=1,
    max_size=50,
)

# ---------------------------------------------------------------------------
# Property 1 — export always returns parseable JSON with required keys
# ---------------------------------------------------------------------------


@given(
    session=st.one_of(st.none(), _printable_text),
    since=st.one_of(st.none(), _printable_text),
)
@settings(max_examples=30, deadline=5000)
def test_export_json_always_valid_json(
    session: str | None,
    since: str | None,
) -> None:
    """For any session/since input, export_audit always returns parseable JSON."""
    db = _fresh_db()

    async def _run() -> str:
        async with aiosqlite.connect(db) as conn:
            await conn.execute(_CALL_LOG_DDL)
            await conn.execute(_TEAM_RUNS_DDL)
            await conn.execute(_SPECIALIST_RUNS_DDL)
            await conn.commit()
        return await export_audit([db], session=session, since=since, fmt="json")

    raw = asyncio.run(_run())

    # Must always be valid JSON
    data = json.loads(raw)

    # Must always have all four required top-level keys
    assert "generated_at" in data
    assert "filters" in data
    assert "call_log" in data
    assert "team_runs" in data

    # call_log and team_runs must always be lists
    assert isinstance(data["call_log"], list)
    assert isinstance(data["team_runs"], list)


# ---------------------------------------------------------------------------
# Property 2 — filters field always matches inputs exactly
# ---------------------------------------------------------------------------


@given(
    session=st.one_of(st.none(), _printable_text),
    since=st.one_of(st.none(), _printable_text),
)
@settings(max_examples=30, deadline=5000)
def test_export_json_filters_field_matches_input(
    session: str | None,
    since: str | None,
) -> None:
    """The filters field in output always echoes the inputs exactly."""
    db = _fresh_db()

    async def _run() -> str:
        async with aiosqlite.connect(db) as conn:
            await conn.execute(_CALL_LOG_DDL)
            await conn.execute(_TEAM_RUNS_DDL)
            await conn.execute(_SPECIALIST_RUNS_DDL)
            await conn.commit()
        return await export_audit([db], session=session, since=since, fmt="json")

    raw = asyncio.run(_run())
    data = json.loads(raw)

    assert data["filters"]["session"] == session
    assert data["filters"]["since"] == since


# ---------------------------------------------------------------------------
# Property 3 — session filter never leaks rows from other sessions
# ---------------------------------------------------------------------------


@given(
    session_a=_printable_text,
    session_b=_printable_text,
)
@settings(max_examples=25, deadline=5000)
def test_export_call_log_session_filter_never_leaks(
    session_a: str,
    session_b: str,
) -> None:
    """Filtering by session_a must never return rows belonging to session_b."""
    # When the two sessions happen to be identical the filter has nothing to exclude — skip.
    if session_a == session_b:
        return

    db = _fresh_db()

    async def _run() -> str:
        async with aiosqlite.connect(db) as conn:
            await conn.execute(_CALL_LOG_DDL)
            await conn.execute(
                "insert into call_log "
                "(session_id, skill, model, latency_ms, "
                "input_tokens, output_tokens, success, created_at) "
                "values (?,?,?,?,?,?,?,?)",
                (session_a, "skill-a", "model-a", 1.0, 1, 1, 1, "2026-01-15T09:00:00+00:00"),
            )
            await conn.execute(
                "insert into call_log "
                "(session_id, skill, model, latency_ms, "
                "input_tokens, output_tokens, success, created_at) "
                "values (?,?,?,?,?,?,?,?)",
                (session_b, "skill-b", "model-b", 2.0, 2, 2, 1, "2026-01-15T09:01:00+00:00"),
            )
            await conn.commit()
        return await export_audit([db], session=session_a, fmt="json")

    raw = asyncio.run(_run())
    data = json.loads(raw)

    # Every returned row must belong to session_a
    for entry in data["call_log"]:
        assert entry["session_id"] == session_a, (
            f"session_b row leaked into results: {entry!r}"
        )


# ---------------------------------------------------------------------------
# Property 4 — both fmt values always succeed and return non-empty strings
# ---------------------------------------------------------------------------


@given(fmt=st.sampled_from(["json", "csv"]))
@settings(max_examples=20, deadline=5000)
def test_export_fmt_json_or_csv_always_succeeds(
    fmt: str,
) -> None:
    """export_audit with fmt in ['json','csv'] always returns a non-empty string."""
    db = _fresh_db()

    async def _run() -> str:
        async with aiosqlite.connect(db) as conn:
            await conn.execute(_CALL_LOG_DDL)
            await conn.execute(_TEAM_RUNS_DDL)
            await conn.execute(_SPECIALIST_RUNS_DDL)
            # Seed one row so the output is non-trivially exercised
            await conn.execute(
                "insert into call_log "
                "(session_id, skill, model, latency_ms, "
                "input_tokens, output_tokens, success, created_at) "
                "values (?,?,?,?,?,?,?,?)",
                ("s1", "sk", "m", 1.0, 1, 1, 1, "2026-01-15T09:00:00+00:00"),
            )
            await conn.commit()
        return await export_audit([db], fmt=fmt)

    result = asyncio.run(_run())

    assert isinstance(result, str)
    assert len(result) > 0
