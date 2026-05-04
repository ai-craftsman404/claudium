"""Tests for configuration loading."""

from pathlib import Path

from claudium.config import load_config


def test_defaults_when_no_toml(tmp_path: Path) -> None:
    config = load_config(tmp_path / "nonexistent.toml")
    assert config.model == "claude-opus-4-5"
    assert config.sandbox == "virtual"
    assert config.typed_retries == 3


def test_loads_toml(tmp_path: Path) -> None:
    toml = tmp_path / "claudium.toml"
    toml.write_text(
        '[agent]\nmodel = "claude-haiku-4-5"\n\n[sandbox]\nallow_shell = false\n',
        encoding="utf-8",
    )
    config = load_config(toml)
    assert config.model == "claude-haiku-4-5"
