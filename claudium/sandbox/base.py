"""Sandbox provider interface and policy enforcement."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SandboxPolicy:
    allow_write: bool = False
    allow_shell: bool = False
    allowed_commands: tuple[str, ...] = ()
    allow_compound_commands: bool = False


@dataclass(frozen=True)
class SandboxFileInfo:
    path: str
    is_dir: bool = False
    size: int = 0


class SandboxBackend(Protocol):
    provider: str
    policy: SandboxPolicy

    @property
    def id(self) -> str: ...
    def list_files(self, path: str = ".") -> list[SandboxFileInfo]: ...
    def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> str: ...
    def write_file(self, path: str, content: str) -> str: ...
    def edit_file(self, path: str, old: str, new: str, *, replace_all: bool = False) -> str: ...
    def grep(self, pattern: str, *, path: str = ".", include: str | None = None) -> str: ...
    def glob(self, pattern: str) -> str: ...
    def shell(self, command: str, *, timeout: int | None = 120) -> dict[str, Any]: ...


def require_write(policy: SandboxPolicy) -> None:
    if not policy.allow_write:
        raise PermissionError("Write access is disabled for this sandbox.")


def require_shell(policy: SandboxPolicy, command: str) -> None:
    if not policy.allow_shell:
        raise PermissionError("Shell execution is disabled for this sandbox.")
    _reject_unsafe_shell(command, policy)
    if policy.allowed_commands:
        executable = _first_executable(command)
        if executable not in policy.allowed_commands:
            raise PermissionError(f"Command not in allowlist: {executable}")


def _reject_unsafe_shell(command: str, policy: SandboxPolicy) -> None:
    if policy.allow_compound_commands:
        return
    if "`" in command or "$(" in command or "\n" in command:
        raise PermissionError("Compound shell syntax is disabled.")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise PermissionError(f"Invalid shell command: {exc}") from exc
    banned = {"&&", "||", ";", "|", ">", "<", ">>", "<<"}
    for token in tokens:
        if token in banned or set(token) <= {"&", "|", ";", ">", "<"}:
            raise PermissionError("Compound shell syntax is disabled.")


def _first_executable(command: str) -> str:
    try:
        parts = shlex.split(command)
        return parts[0] if parts else ""
    except ValueError:
        return ""
