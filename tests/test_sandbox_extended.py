"""Extended sandbox tests — edit, list, glob, grep, offset reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from claudium.sandbox.base import SandboxPolicy
from claudium.sandbox.virtual import VirtualSandbox


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def rw(root: Path) -> VirtualSandbox:
    return VirtualSandbox(root, SandboxPolicy(allow_write=True, allow_shell=True))


@pytest.fixture
def ro(root: Path) -> VirtualSandbox:
    return VirtualSandbox(root, SandboxPolicy())


# ── read_file ──────────────────────────────────────────────────────────────────

def test_read_file_full(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("a\nb\nc\nd\ne", encoding="utf-8")
    assert rw.read_file("f.txt") == "a\nb\nc\nd\ne"


def test_read_file_with_offset(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("a\nb\nc\nd\ne", encoding="utf-8")
    assert rw.read_file("f.txt", offset=3) == "c\nd\ne"


def test_read_file_with_limit(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("a\nb\nc\nd\ne", encoding="utf-8")
    assert rw.read_file("f.txt", limit=2) == "a\nb"


def test_read_file_offset_and_limit(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("a\nb\nc\nd\ne", encoding="utf-8")
    assert rw.read_file("f.txt", offset=2, limit=2) == "b\nc"


# ── edit_file ──────────────────────────────────────────────────────────────────

def test_edit_file_replaces_first_occurrence(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("hello world hello", encoding="utf-8")
    rw.edit_file("f.txt", "hello", "hi")
    assert (root / "f.txt").read_text() == "hi world hello"


def test_edit_file_replace_all(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("hello world hello", encoding="utf-8")
    rw.edit_file("f.txt", "hello", "hi", replace_all=True)
    assert (root / "f.txt").read_text() == "hi world hi"


def test_edit_file_raises_if_string_not_found(rw: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("hello world", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        rw.edit_file("f.txt", "missing", "replacement")


def test_edit_file_blocked_without_write(ro: VirtualSandbox, root: Path) -> None:
    (root / "f.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(PermissionError):
        ro.edit_file("f.txt", "hello", "hi")


# ── list_files ─────────────────────────────────────────────────────────────────

def test_list_files_returns_entries(rw: VirtualSandbox, root: Path) -> None:
    (root / "a.py").write_text("", encoding="utf-8")
    (root / "b.py").write_text("", encoding="utf-8")
    paths = [f.path for f in rw.list_files()]
    assert "a.py" in paths
    assert "b.py" in paths


def test_list_files_marks_directories(rw: VirtualSandbox, root: Path) -> None:
    (root / "subdir").mkdir()
    entries = {f.path: f for f in rw.list_files()}
    assert entries["subdir"].is_dir is True


def test_list_files_reports_size(rw: VirtualSandbox, root: Path) -> None:
    (root / "sized.txt").write_text("hello", encoding="utf-8")
    entries = {f.path: f for f in rw.list_files()}
    assert entries["sized.txt"].size == 5


# ── glob ───────────────────────────────────────────────────────────────────────

def test_glob_finds_matching_files(rw: VirtualSandbox, root: Path) -> None:
    (root / "a.py").write_text("", encoding="utf-8")
    (root / "b.txt").write_text("", encoding="utf-8")
    result = rw.glob("*.py")
    assert "a.py" in result
    assert "b.txt" not in result


def test_glob_no_matches(rw: VirtualSandbox) -> None:
    result = rw.glob("*.nonexistent")
    assert result == "(no matches)"


# ── grep ───────────────────────────────────────────────────────────────────────

def test_grep_finds_pattern(rw: VirtualSandbox, root: Path) -> None:
    (root / "code.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    result = rw.grep("def hello")
    assert "hello" in result


def test_grep_no_match(rw: VirtualSandbox, root: Path) -> None:
    (root / "code.py").write_text("nothing here\n", encoding="utf-8")
    result = rw.grep("def missing")
    assert result == "(no matches)"


# ── shell ──────────────────────────────────────────────────────────────────────

def test_shell_returns_stdout(rw: VirtualSandbox) -> None:
    result = rw.shell("echo claudium")
    assert "claudium" in result["stdout"]
    assert result["returncode"] == 0


def test_shell_captures_stderr(rw: VirtualSandbox) -> None:
    result = rw.shell("python3 -c \"import sys; sys.stderr.write('err\\n')\"")
    assert "err" in result["stderr"]


def test_shell_with_allowlist_blocks_unlisted(root: Path) -> None:
    from claudium.sandbox.base import SandboxPolicy
    sb = VirtualSandbox(root, SandboxPolicy(allow_shell=True, allowed_commands=("echo",)))
    with pytest.raises(PermissionError, match="not in allowlist"):
        sb.shell("ls .")


def test_shell_with_allowlist_permits_listed(root: Path) -> None:
    from claudium.sandbox.base import SandboxPolicy
    sb = VirtualSandbox(root, SandboxPolicy(allow_shell=True, allowed_commands=("echo",)))
    result = sb.shell("echo ok")
    assert "ok" in result["stdout"]
