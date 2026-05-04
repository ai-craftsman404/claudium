"""Markdown skill and role discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import frontmatter

from claudium.types import Role, Skill


def load_skills(root: str | Path, skills_dir: str | Path | None = None) -> dict[str, Skill]:
    base = Path(root).expanduser().resolve()
    directory = Path(skills_dir).expanduser() if skills_dir else base / ".agents" / "skills"
    if not directory.is_absolute():
        directory = base / directory
    if not directory.exists():
        return {}
    return {s.name: s for s in (parse_skill(p) for p in sorted(directory.rglob("*.md")))}


def load_roles(root: str | Path, roles_dir: str | Path | None = None) -> dict[str, Role]:
    base = Path(root).expanduser().resolve()
    directory = Path(roles_dir).expanduser() if roles_dir else base / ".agents" / "roles"
    if not directory.is_absolute():
        directory = base / directory
    if not directory.exists():
        return {}
    return {r.name: r for r in (parse_role(p) for p in sorted(directory.rglob("*.md")))}


def load_project_instructions(root: str | Path) -> str:
    base = Path(root).expanduser().resolve()
    parts = []
    for filename in ["CLAUDE.md"]:
        path = base / filename
        if path.exists():
            parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(p for p in parts if p)


def parse_skill(path: str | Path) -> Skill:
    p = Path(path).expanduser().resolve()
    post = frontmatter.loads(p.read_text(encoding="utf-8"))
    meta: dict[str, Any] = dict(post.metadata or {})
    default_name = p.parent.name if p.name.upper() == "SKILL.MD" else p.stem
    name = str(meta.get("name") or default_name).strip()
    if not name:
        raise ValueError(f"Skill name cannot be empty: {p}")
    return Skill(
        name=name,
        description=str(meta.get("description", "") or "").strip(),
        instructions=str(post.content or "").strip(),
        input_schema=_schema_or_none(meta.get("input_schema")),
        output_schema=_schema_or_none(meta.get("output_schema")),
        path=p,
    )


def parse_role(path: str | Path) -> Role:
    p = Path(path).expanduser().resolve()
    post = frontmatter.loads(p.read_text(encoding="utf-8"))
    meta: dict[str, Any] = dict(post.metadata or {})
    name = str(meta.get("name") or p.stem).strip()
    if not name:
        raise ValueError(f"Role name cannot be empty: {p}")
    return Role(
        name=name,
        description=str(meta.get("description", "") or "").strip(),
        instructions=str(post.content or "").strip(),
        model=str(meta["model"]).strip() if meta.get("model") else None,
        path=p,
    )


def render_skill_prompt(skill: Skill, args: dict[str, Any] | None = None) -> str:
    parts = [skill.instructions]
    if args:
        parts.append("Arguments:\n" + json.dumps(args, indent=2, sort_keys=True))
    return "\n\n".join(p for p in parts if p).strip()


def _schema_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
