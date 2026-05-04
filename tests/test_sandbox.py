"""Tests for sandbox policy enforcement."""

from pathlib import Path

import pytest

from claudium.sandbox.base import SandboxPolicy
from claudium.sandbox.virtual import VirtualSandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> VirtualSandbox:
    return VirtualSandbox(tmp_path, SandboxPolicy())


@pytest.fixture
def writable_sandbox(tmp_path: Path) -> VirtualSandbox:
    return VirtualSandbox(tmp_path, SandboxPolicy(allow_write=True))


def test_read_file(writable_sandbox: VirtualSandbox, tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3")
    content = writable_sandbox.read_file("hello.txt")
    assert "line1" in content


def test_write_blocked_by_default(sandbox: VirtualSandbox) -> None:
    with pytest.raises(PermissionError, match="Write access"):
        sandbox.write_file("out.txt", "hello")


def test_write_allowed(writable_sandbox: VirtualSandbox) -> None:
    writable_sandbox.write_file("out.txt", "hello")
    assert (writable_sandbox.root / "out.txt").read_text() == "hello"


def test_shell_blocked_by_default(sandbox: VirtualSandbox) -> None:
    with pytest.raises(PermissionError, match="Shell execution"):
        sandbox.shell("echo hello")


def test_shell_compound_blocked(tmp_path: Path) -> None:
    sb = VirtualSandbox(tmp_path, SandboxPolicy(allow_shell=True))
    with pytest.raises(PermissionError, match="Compound"):
        sb.shell("echo hello && echo world")


def test_shell_allowlist(tmp_path: Path) -> None:
    sb = VirtualSandbox(
        tmp_path,
        SandboxPolicy(allow_shell=True, allowed_commands=("git",)),
    )
    with pytest.raises(PermissionError, match="not in allowlist"):
        sb.shell("rm -rf .")
