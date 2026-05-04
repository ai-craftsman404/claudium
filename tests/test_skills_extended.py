"""Extended skills tests — directory loading, project instructions, edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from claudium.skills import (
    load_project_instructions,
    load_roles,
    load_skills,
    parse_skill,
    render_skill_prompt,
)


# ── load_skills ────────────────────────────────────────────────────────────────

def test_load_skills_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    skills = load_skills(tmp_path)
    assert skills == {}


def test_load_skills_finds_all_md_files(tmp_path: Path) -> None:
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "triage.md").write_text("---\nname: triage\n---\nTriage.\n", encoding="utf-8")
    (skills_dir / "summary.md").write_text("---\nname: summary\n---\nSummarise.\n", encoding="utf-8")
    skills = load_skills(tmp_path)
    assert set(skills.keys()) == {"triage", "summary"}


def test_load_skills_uses_stem_as_default_name(tmp_path: Path) -> None:
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "myskill.md").write_text("---\n---\nDo something.\n", encoding="utf-8")
    skills = load_skills(tmp_path)
    assert "myskill" in skills


def test_load_skills_custom_dir(tmp_path: Path) -> None:
    custom = tmp_path / "custom_skills"
    custom.mkdir()
    (custom / "foo.md").write_text("---\nname: foo\n---\nFoo.\n", encoding="utf-8")
    skills = load_skills(tmp_path, skills_dir=custom)
    assert "foo" in skills


def test_skill_md_uses_parent_dir_as_name(tmp_path: Path) -> None:
    # A file named skill.md (p.name.upper() == "SKILL.MD") falls back to its parent dir name
    skills_dir = tmp_path / ".agents" / "skills" / "triage"
    skills_dir.mkdir(parents=True)
    (skills_dir / "skill.md").write_text("---\n---\nDo triage.\n", encoding="utf-8")
    skills = load_skills(tmp_path)
    assert "triage" in skills


# ── load_roles ─────────────────────────────────────────────────────────────────

def test_load_roles_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    roles = load_roles(tmp_path)
    assert roles == {}


def test_load_roles_parses_model_field(tmp_path: Path) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "fast.md").write_text(
        "---\nname: fast\nmodel: claude-haiku-4-5\n---\nBe fast.\n", encoding="utf-8"
    )
    roles = load_roles(tmp_path)
    assert roles["fast"].model == "claude-haiku-4-5"


def test_load_roles_model_is_none_when_absent(tmp_path: Path) -> None:
    roles_dir = tmp_path / ".agents" / "roles"
    roles_dir.mkdir(parents=True)
    (roles_dir / "plain.md").write_text("---\nname: plain\n---\nContent.\n", encoding="utf-8")
    roles = load_roles(tmp_path)
    assert roles["plain"].model is None


# ── load_project_instructions ──────────────────────────────────────────────────

def test_load_project_instructions_reads_claude_md(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Project\n\nDo good things.", encoding="utf-8")
    result = load_project_instructions(tmp_path)
    assert "Do good things" in result


def test_load_project_instructions_empty_when_missing(tmp_path: Path) -> None:
    result = load_project_instructions(tmp_path)
    assert result == ""


def test_load_project_instructions_strips_whitespace(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("  \n\n# Title\n\nContent.\n\n  ", encoding="utf-8")
    result = load_project_instructions(tmp_path)
    assert result == result.strip()


# ── render_skill_prompt ────────────────────────────────────────────────────────

def test_render_includes_instructions(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text("---\nname: s\n---\nDo the thing.\n", encoding="utf-8")
    skill = parse_skill(p)
    assert "Do the thing" in render_skill_prompt(skill)


def test_render_includes_args_as_json(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text("---\nname: s\n---\nInstructions.\n", encoding="utf-8")
    skill = parse_skill(p)
    rendered = render_skill_prompt(skill, args={"issue": 42, "repo": "org/repo"})
    assert '"issue": 42' in rendered
    assert '"repo": "org/repo"' in rendered


def test_render_without_args_omits_arguments_block(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text("---\nname: s\n---\nInstructions.\n", encoding="utf-8")
    skill = parse_skill(p)
    rendered = render_skill_prompt(skill)
    assert "Arguments" not in rendered
