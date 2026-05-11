"""Audit log export — queries call_log, team_runs_v3, and specialist_runs for compliance reports."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass
class CallLogEntry:
    session_id: str
    skill: str | None
    model: str | None
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    success: int
    created_at: str
    user_id: str | None = None


@dataclass
class SpecialistRunEntry:
    specialist_name: str
    output_text: str
    fitness_score: float


@dataclass
class TeamRunEntry:
    run_id: str
    prompt: str
    domain: str
    specialists_used: list[str]
    adjudication: dict[str, Any] | None
    synthesis: str | None
    created_at: str
    specialist_runs: list[SpecialistRunEntry] = field(default_factory=list)


@dataclass
class AuditReport:
    generated_at: str
    filters: dict[str, Any]
    call_log: list[CallLogEntry]
    team_runs: list[TeamRunEntry]
    budget_consumed: int = 0
    budget_limit: int | None = None
    user_id: str | None = None


async def _query_call_log(
    db: aiosqlite.Connection,
    *,
    session: str | None,
    since: str | None,
    user_id: str | None = None,
) -> list[CallLogEntry]:
    # Check which columns exist to support old DBs without user_id
    try:
        col_cursor = await db.execute("PRAGMA table_info(call_log)")
        col_names = {row[1] for row in await col_cursor.fetchall()}
    except Exception:
        return []
    has_user_id = "user_id" in col_names

    clauses: list[str] = []
    params: list[Any] = []
    if session:
        clauses.append("session_id = ?")
        params.append(session)
    if since:
        clauses.append("created_at >= ?")
        params.append(since)
    if user_id is not None and has_user_id:
        clauses.append("COALESCE(user_id, '') = ?")
        params.append(user_id)
    elif user_id is not None and not has_user_id:
        # No user_id column — no rows can match
        return []
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    select_cols = (
        "session_id, skill, model, latency_ms, input_tokens, output_tokens, "
        "success, created_at" + (", user_id" if has_user_id else "")
    )
    try:
        cursor = await db.execute(
            f"SELECT {select_cols} FROM call_log {where} ORDER BY id ASC",
            params,
        )
        rows = await cursor.fetchall()
    except Exception:
        return []
    return [
        CallLogEntry(
            session_id=r[0], skill=r[1], model=r[2], latency_ms=r[3],
            input_tokens=r[4], output_tokens=r[5], success=r[6], created_at=r[7],
            user_id=r[8] if has_user_id and len(r) > 8 else None,
        )
        for r in rows
    ]


async def _query_team_runs(
    db: aiosqlite.Connection,
    *,
    since: str | None,
) -> list[TeamRunEntry]:
    where = "WHERE created_at >= ?" if since else ""
    params: list[Any] = [since] if since else []
    try:
        cursor = await db.execute(
            f"SELECT id, prompt, domain, specialists_used, adjudication, synthesis, created_at "
            f"FROM team_runs_v3 {where} ORDER BY created_at ASC",
            params,
        )
        rows = await cursor.fetchall()
    except Exception:
        return []

    entries: list[TeamRunEntry] = []
    for r in rows:
        entry = TeamRunEntry(
            run_id=r[0],
            prompt=r[1],
            domain=r[2],
            specialists_used=json.loads(r[3]) if r[3] else [],
            adjudication=json.loads(r[4]) if r[4] else None,
            synthesis=r[5],
            created_at=r[6],
        )
        try:
            sr_cursor = await db.execute(
                "SELECT specialist_name, output_text, fitness_score "
                "FROM specialist_runs WHERE run_id = ? ORDER BY id ASC",
                (r[0],),
            )
            entry.specialist_runs = [
                SpecialistRunEntry(specialist_name=sr[0], output_text=sr[1], fitness_score=sr[2])
                for sr in await sr_cursor.fetchall()
            ]
        except Exception:
            pass
        entries.append(entry)
    return entries


async def export_audit(
    db_paths: list[Path],
    *,
    session: str | None = None,
    since: str | None = None,
    fmt: str = "json",
    budget_limit: int | None = None,
    user_id: str | None = None,
) -> str:
    """Query all db_paths and return a compliance audit report as JSON or CSV."""
    call_log: list[CallLogEntry] = []
    team_runs: list[TeamRunEntry] = []

    for db_path in db_paths:
        if not db_path.exists():
            continue
        # team_runs_v3 has no session_id column; scope by db filename (= session_id)
        include_team_runs = session is None or db_path.stem == session
        async with aiosqlite.connect(db_path) as db:
            call_log.extend(
                await _query_call_log(db, session=session, since=since, user_id=user_id)
            )
            if include_team_runs:
                team_runs.extend(await _query_team_runs(db, since=since))

    budget_consumed = sum(
        (e.input_tokens or 0) + (e.output_tokens or 0) for e in call_log
    )
    report = AuditReport(
        generated_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        filters={"session": session, "since": since, "user_id": user_id},
        call_log=call_log,
        team_runs=team_runs,
        budget_consumed=budget_consumed,
        budget_limit=budget_limit,
        user_id=user_id,
    )

    return _to_csv(report) if fmt == "csv" else _to_json(report)


def _to_json(report: AuditReport) -> str:
    def _serial(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serial(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_serial(i) for i in obj]
        return obj

    return json.dumps(_serial(report), indent=2)


def _to_csv(report: AuditReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["# CALL LOG"])
    writer.writerow([
        "session_id", "skill", "model", "latency_ms",
        "input_tokens", "output_tokens", "success", "created_at",
    ])
    for e in report.call_log:
        writer.writerow([
            e.session_id, e.skill or "", e.model or "",
            e.latency_ms if e.latency_ms is not None else "",
            e.input_tokens if e.input_tokens is not None else "",
            e.output_tokens if e.output_tokens is not None else "",
            e.success, e.created_at,
        ])

    writer.writerow([])
    writer.writerow(["# TEAM RUNS V3"])
    writer.writerow([
        "run_id", "domain", "prompt", "specialists_used",
        "adjudication_mode", "adjudication_accepted", "synthesis", "created_at",
    ])
    for t in report.team_runs:
        adj_mode = t.adjudication.get("mode", "") if t.adjudication else ""
        adj_accepted = t.adjudication.get("accepted", "") if t.adjudication else ""
        writer.writerow([
            t.run_id, t.domain, t.prompt,
            ";".join(t.specialists_used),
            adj_mode, adj_accepted, t.synthesis or "", t.created_at,
        ])

    return buf.getvalue()
