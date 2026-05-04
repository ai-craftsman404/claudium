"""Tests for deployment file generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from claudium.build import build


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\nname = "my-agent"\n', encoding="utf-8"
    )
    return tmp_path


def _paths(files: list) -> set[str]:
    return {f[0].name for f in files}


# ── targets ────────────────────────────────────────────────────────────────────

def test_docker_generates_dockerfile(root: Path) -> None:
    files = build("docker", root)
    assert "Dockerfile" in _paths(files)


def test_docker_generates_dockerignore(root: Path) -> None:
    files = build("docker", root)
    assert ".dockerignore" in _paths(files)


def test_railway_generates_railway_toml(root: Path) -> None:
    files = build("railway", root)
    assert "railway.toml" in _paths(files)


def test_fly_generates_fly_toml(root: Path) -> None:
    files = build("fly", root)
    assert "fly.toml" in _paths(files)


def test_fly_uses_app_name_from_toml(root: Path) -> None:
    files = build("fly", root)
    fly_content = next(c for p, c in files if p.name == "fly.toml")
    assert "my-agent" in fly_content


def test_render_generates_render_yaml(root: Path) -> None:
    files = build("render", root)
    assert "render.yaml" in _paths(files)


def test_ci_generates_github_workflow(root: Path) -> None:
    files = build("ci", root)
    assert "ci.yml" in _paths(files)


def test_unknown_target_raises(root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        build("heroku", root)


# ── file writing ───────────────────────────────────────────────────────────────

def test_build_writes_files(root: Path) -> None:
    files = build("docker", root)
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert (root / "Dockerfile").exists()


def test_dockerfile_contains_claudium_dev(root: Path) -> None:
    files = build("docker", root)
    dockerfile = next(c for p, c in files if p.name == "Dockerfile")
    assert "claudium" in dockerfile
    assert "2024" in dockerfile


def test_app_name_falls_back_to_dir_name(tmp_path: Path) -> None:
    files = build("fly", tmp_path)
    fly_content = next(c for p, c in files if p.name == "fly.toml")
    assert tmp_path.name in fly_content
