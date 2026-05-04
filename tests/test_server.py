"""Tests for the Claudium webhook server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claudium.server import AgentContext, create_app


@pytest.fixture
def agent_root(tmp_path: Path) -> Path:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def app_with_agent(agent_root: Path):
    (agent_root / "agents" / "triage.py").write_text(
        'triggers = {"webhook": True}\n\n'
        "async def triage(context):\n"
        '    return {"received": context.payload}\n',
        encoding="utf-8",
    )
    return create_app(agent_root)


@pytest.fixture
def app_no_agents(agent_root: Path):
    return create_app(agent_root)


# ── health ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok(app_with_agent) -> None:
    with TestClient(app_with_agent) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_lists_agents(app_with_agent) -> None:
    with TestClient(app_with_agent) as client:
        resp = client.get("/health")
    assert "triage" in resp.json()["agents"]


# ── webhook routing ────────────────────────────────────────────────────────────

def test_agent_receives_payload(app_with_agent) -> None:
    with TestClient(app_with_agent) as client:
        resp = client.post(
            "/agents/triage/run-1",
            json={"payload": {"issue_number": 42}},
        )
    assert resp.status_code == 200
    assert resp.json()["received"]["issue_number"] == 42


def test_unknown_agent_returns_404(app_with_agent) -> None:
    with TestClient(app_with_agent) as client:
        resp = client.post("/agents/nonexistent/run-1", json={})
    assert resp.status_code == 404


def test_agent_without_webhook_trigger_not_exposed(agent_root: Path) -> None:
    (agent_root / "agents" / "hidden.py").write_text(
        "triggers = {}\n\nasync def hidden(context):\n    return {}\n",
        encoding="utf-8",
    )
    app = create_app(agent_root)
    with TestClient(app) as client:
        resp = client.post("/agents/hidden/run-1", json={})
    assert resp.status_code == 404


def test_payload_unwrapped_from_body(app_with_agent) -> None:
    with TestClient(app_with_agent) as client:
        resp = client.post(
            "/agents/triage/run-1",
            json={"payload": {"key": "value"}},
        )
    assert resp.json()["received"]["key"] == "value"


def test_agent_id_passed_to_context(agent_root: Path) -> None:
    (agent_root / "agents" / "echo.py").write_text(
        'triggers = {"webhook": True}\n\n'
        "async def echo(context):\n"
        '    return {"agent_id": context.agent_id}\n',
        encoding="utf-8",
    )
    app = create_app(agent_root)
    with TestClient(app) as client:
        resp = client.post("/agents/echo/my-session-id", json={})
    assert resp.json()["agent_id"] == "my-session-id"


def test_agent_exception_returns_500(agent_root: Path) -> None:
    (agent_root / "agents" / "broken.py").write_text(
        'triggers = {"webhook": True}\n\n'
        "async def broken(context):\n"
        '    raise RuntimeError("something went wrong")\n',
        encoding="utf-8",
    )
    app = create_app(agent_root)
    with TestClient(app) as client:
        resp = client.post("/agents/broken/run-1", json={})
    assert resp.status_code == 500


# ── AgentContext ───────────────────────────────────────────────────────────────

def test_agent_context_exposes_payload(tmp_path: Path) -> None:
    ctx = AgentContext(agent_id="abc", payload={"x": 1}, root=tmp_path)
    assert ctx.payload == {"x": 1}
    assert ctx.agent_id == "abc"
