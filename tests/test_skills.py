"""Tests for skill and role loading."""

from pathlib import Path

import pytest

from claudium.skills import parse_role, parse_skill, render_skill_prompt


@pytest.fixture
def skill_file(tmp_path: Path) -> Path:
    p = tmp_path / "triage.md"
    p.write_text(
        "---\nname: triage\ndescription: Classify an issue\n---\n\nTriage the issue.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def role_file(tmp_path: Path) -> Path:
    p = tmp_path / "analyst.md"
    p.write_text(
        "---\nname: analyst\nmodel: claude-sonnet-4-5\n---\n\nBe concise.\n",
        encoding="utf-8",
    )
    return p


def test_parse_skill(skill_file: Path) -> None:
    skill = parse_skill(skill_file)
    assert skill.name == "triage"
    assert skill.description == "Classify an issue"
    assert "Triage" in skill.instructions


def test_parse_role(role_file: Path) -> None:
    role = parse_role(role_file)
    assert role.name == "analyst"
    assert role.model == "claude-sonnet-4-5"


def test_render_skill_prompt(skill_file: Path) -> None:
    skill = parse_skill(skill_file)
    rendered = render_skill_prompt(skill, args={"issue_number": 42})
    assert "42" in rendered
    assert "Triage" in rendered
