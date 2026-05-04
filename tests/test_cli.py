"""Tests for the Claudium CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from claudium.cli import app

runner = CliRunner()


# ── claudium init ──────────────────────────────────────────────────────────────

def test_init_creates_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path / "my-agent")])
    assert result.exit_code == 0
    assert (tmp_path / "my-agent").is_dir()


def test_init_creates_expected_structure(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    runner.invoke(app, ["init", str(project)])
    assert (project / "claudium.toml").exists()
    assert (project / "CLAUDE.md").exists()
    assert (project / ".agents" / "skills").is_dir()
    assert (project / ".agents" / "roles").is_dir()
    assert (project / "agents").is_dir()
    assert (project / "agents" / "default.py").exists()
    assert (project / ".gitignore").exists()


def test_init_creates_valid_toml(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    runner.invoke(app, ["init", str(project)])
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with (project / "claudium.toml").open("rb") as f:
        config = tomllib.load(f)
    assert config["agent"]["model"] == "claude-opus-4-5"


def test_init_fails_if_directory_exists(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    result = runner.invoke(app, ["init", str(existing)])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_init_creates_default_skill(tmp_path: Path) -> None:
    project = tmp_path / "my-agent"
    runner.invoke(app, ["init", str(project)])
    assert (project / ".agents" / "skills" / "default.md").exists()


# ── claudium run ───────────────────────────────────────────────────────────────

def _make_mock_response(text: str) -> MagicMock:
    """Build a mock that matches anthropic.types.Message structure."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock()
    response.usage.model_dump.return_value = {}
    return response


@pytest.fixture
def mock_anthropic(tmp_path: Path):
    """Patch AsyncAnthropic so claudium run never hits the network."""
    response = _make_mock_response("triage complete")
    with patch("anthropic.AsyncAnthropic") as mock_cls:
        instance = MagicMock()
        instance.messages.create = AsyncMock(return_value=response)
        mock_cls.return_value = instance
        yield mock_cls


def test_run_invokes_prompt(tmp_path: Path, mock_anthropic, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8"
    )
    result = runner.invoke(app, ["run", "--prompt", "hello"])
    assert result.exit_code == 0
    assert "triage complete" in result.output


def test_run_uses_model_override(tmp_path: Path, mock_anthropic, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8"
    )
    runner.invoke(app, ["run", "--prompt", "hello", "--model", "claude-haiku-4-5"])
    call_kwargs = mock_anthropic.return_value.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5"
