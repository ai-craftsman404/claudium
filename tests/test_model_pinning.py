"""TDD tests for v3d model version pinning — written before implementation.

Covers:
- HarnessResult.model field defaults and storage
- MockHarness returning model=None vs explicit model strings
- call_log writing result.model (or falling back to config.model)
- per-session model override stored in call_log
- ClaudiumConfig.pinned_model field and TOML loading
- model precedence: session override > config.pinned_model
- call_log schema includes model column
- per-call (not per-session) model granularity
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from claudium.config import load_config
from claudium.core import ClaudiumAgent
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult

# ── MockHarness ───────────────────────────────────────────────────────────────


class MockHarness:
    """Minimal harness that returns a canned text and optional model string."""

    def __init__(
        self,
        response_text: str = "ok",
        response_model: str | None = None,
    ) -> None:
        self._text = response_text
        self._model = response_model

    async def run(self, *, prompt, system_prompt, config, **_) -> HarnessResult:
        return HarnessResult(text=self._text, model=self._model)

    async def stream(
        self, *, prompt, system_prompt, config, **_
    ) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": self._text})


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _make_session(
    tmp_path: Path,
    *,
    harness: MockHarness,
    config_model: str = "claude-opus-4-5",
    pinned_model: str | None = None,
    session_model: str | None = None,
):
    """Build a ClaudiumAgent + session with the given harness and config."""
    cfg = ClaudiumConfig(root=tmp_path, model=config_model, pinned_model=pinned_model)
    (tmp_path / ".claudium" / "sessions").mkdir(parents=True, exist_ok=True)
    agent = ClaudiumAgent(config=cfg, harness=harness)
    session = await agent.session("test-session", model=session_model)
    return agent, session


async def _read_call_log_models(db_path: Path) -> list[str | None]:
    """Return the model column values from call_log, ordered by id."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("select model from call_log order by id")
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


# ── 1. HarnessResult default ──────────────────────────────────────────────────


def test_harness_result_model_field_default_none() -> None:
    """HarnessResult() has model=None by default."""
    result = HarnessResult(text="hello")
    assert result.model is None


# ── 2. HarnessResult explicit model ──────────────────────────────────────────


def test_harness_result_model_field_settable() -> None:
    """HarnessResult(text=..., model=...) stores the model string correctly."""
    result = HarnessResult(text="hello", model="claude-opus-4-7")
    assert result.model == "claude-opus-4-7"


# ── 3. MockHarness returns model=None ────────────────────────────────────────


@pytest.mark.asyncio
async def test_mock_harness_returns_model_none() -> None:
    """MockHarness without response_model yields HarnessResult.model == None."""
    harness = MockHarness(response_text="answer")
    result = await harness.run(prompt="q", system_prompt="", config=ClaudiumConfig())
    assert result.model is None


# ── 4. call_log records model from harness result ────────────────────────────


@pytest.mark.asyncio
async def test_call_log_records_model_from_harness_result(tmp_path: Path) -> None:
    """When MockHarness returns model='test-model', call_log stores 'test-model'."""
    harness = MockHarness(response_model="test-model")
    _, session = await _make_session(tmp_path, harness=harness)

    await session.prompt("hello")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 1
    assert models[-1] == "test-model"


# ── 5. call_log falls back to config.model when result.model is None ─────────


@pytest.mark.asyncio
async def test_call_log_falls_back_to_config_model_when_result_model_none(
    tmp_path: Path,
) -> None:
    """When harness returns model=None, call_log falls back to config.model."""
    harness = MockHarness(response_model=None)
    _, session = await _make_session(
        tmp_path, harness=harness, config_model="claude-sonnet-4-5"
    )

    await session.prompt("hello")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 1
    assert models[-1] == "claude-sonnet-4-5"


# ── 6. session model override stored in call_log ──────────────────────────────


@pytest.mark.asyncio
async def test_session_model_override_stored_in_call_log(tmp_path: Path) -> None:
    """session(model='override-model') causes call_log to store 'override-model'."""
    harness = MockHarness(response_model=None)  # harness returns no model info
    _, session = await _make_session(
        tmp_path,
        harness=harness,
        config_model="claude-opus-4-5",
        session_model="override-model",
    )

    await session.prompt("hello")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 1
    assert models[-1] == "override-model"


# ── 7. ClaudiumConfig.pinned_model dataclass field ───────────────────────────


def test_pinned_model_in_config_dataclass() -> None:
    """ClaudiumConfig(pinned_model='x').pinned_model == 'x'."""
    cfg = ClaudiumConfig(pinned_model="claude-opus-4-7")
    assert cfg.pinned_model == "claude-opus-4-7"


def test_pinned_model_default_none() -> None:
    """ClaudiumConfig() has pinned_model=None by default."""
    cfg = ClaudiumConfig()
    assert cfg.pinned_model is None


# ── 8. pinned_model loaded from TOML ─────────────────────────────────────────


def test_pinned_model_loaded_from_toml(tmp_path: Path) -> None:
    """[pinning] model = '...' in TOML is read into config.pinned_model."""
    toml = tmp_path / "claudium.toml"
    toml.write_text(
        '[agent]\nmodel = "claude-opus-4-5"\n\n[pinning]\nmodel = "claude-opus-4-7"\n',
        encoding="utf-8",
    )
    cfg = load_config(toml)
    assert cfg.pinned_model == "claude-opus-4-7"


def test_pinned_model_absent_from_toml_gives_none(tmp_path: Path) -> None:
    """TOML without [pinning] section leaves pinned_model=None."""
    toml = tmp_path / "claudium.toml"
    toml.write_text('[agent]\nmodel = "claude-opus-4-5"\n', encoding="utf-8")
    cfg = load_config(toml)
    assert cfg.pinned_model is None


# ── 9. pinned_model used when no session override ─────────────────────────────


@pytest.mark.asyncio
async def test_pinned_model_used_when_no_session_override(tmp_path: Path) -> None:
    """config.pinned_model is used in session when no per-session model override given."""
    harness = MockHarness(response_model=None)
    _, session = await _make_session(
        tmp_path,
        harness=harness,
        config_model="claude-opus-4-5",
        pinned_model="claude-opus-4-7",
        session_model=None,
    )

    await session.prompt("hello")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 1
    # pinned_model should take precedence over the base config.model
    assert models[-1] == "claude-opus-4-7"


# ── 10. model precedence: session override > config pinned_model ──────────────


@pytest.mark.asyncio
async def test_model_precedence_session_over_config(tmp_path: Path) -> None:
    """session(model='A') wins over config(pinned_model='B')."""
    harness = MockHarness(response_model=None)
    _, session = await _make_session(
        tmp_path,
        harness=harness,
        config_model="claude-opus-4-5",
        pinned_model="config-pinned-model",
        session_model="session-override-model",
    )

    await session.prompt("hello")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 1
    assert models[-1] == "session-override-model"


# ── 11. model column present in call_log schema ───────────────────────────────


@pytest.mark.asyncio
async def test_model_column_present_in_call_log_schema(tmp_path: Path) -> None:
    """PRAGMA table_info(call_log) shows a 'model' column."""
    harness = MockHarness()
    _, session = await _make_session(tmp_path, harness=harness)

    async with aiosqlite.connect(session.db_path) as db:
        cursor = await db.execute("pragma table_info(call_log)")
        rows = await cursor.fetchall()

    column_names = [row[1] for row in rows]
    assert "model" in column_names


# ── 12. model recorded per-call, not per-session ─────────────────────────────


@pytest.mark.asyncio
async def test_model_recorded_per_call_not_per_session(tmp_path: Path) -> None:
    """Each call can record a different model value in call_log."""

    class RotatingModelHarness:
        """Returns alternating model strings on successive .run() calls."""

        def __init__(self) -> None:
            self._call_count = 0
            self._models = ["model-alpha", "model-beta"]

        async def run(self, *, prompt, system_prompt, config, **_) -> HarnessResult:
            model = self._models[self._call_count % len(self._models)]
            self._call_count += 1
            return HarnessResult(text="ok", model=model)

        async def stream(
            self, *, prompt, system_prompt, config, **_
        ) -> AsyncIterator[ClaudiumEvent]:
            yield ClaudiumEvent(type="text_delta", data={"text": "ok"})

    harness = RotatingModelHarness()
    cfg = ClaudiumConfig(root=tmp_path)
    (tmp_path / ".claudium" / "sessions").mkdir(parents=True, exist_ok=True)
    agent = ClaudiumAgent(config=cfg, harness=harness)
    session = await agent.session("per-call-test")

    await session.prompt("first call")
    await session.prompt("second call")

    models = await _read_call_log_models(session.db_path)
    assert len(models) >= 2
    assert models[-2] == "model-alpha"
    assert models[-1] == "model-beta"
