"""Virtual (local filesystem) sandbox."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

from claudium.sandbox.base import (
    SandboxFileInfo,
    SandboxPolicy,
    require_shell,
    require_write,
)


class VirtualSandbox:
    provider = "virtual"

    def __init__(self, root: Path, policy: SandboxPolicy) -> None:
        self.root = root.expanduser().resolve()
        self.policy = policy
        self._id = str(uuid.uuid4())

    @property
    def id(self) -> str:
        return self._id

    def list_files(self, path: str = ".") -> list[SandboxFileInfo]:
        target = (self.root / path).resolve()
        return [
            SandboxFileInfo(
                path=str(p.relative_to(self.root)),
                is_dir=p.is_dir(),
                size=p.stat().st_size if p.is_file() else 0,
            )
            for p in sorted(target.iterdir())
        ]

    def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> str:
        lines = (self.root / path).read_text(encoding="utf-8").splitlines()
        start = max(0, offset - 1)
        end = start + limit if limit else None
        return "\n".join(lines[start:end])

    def write_file(self, path: str, content: str) -> str:
        require_write(self.policy)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written: {path}"

    def edit_file(self, path: str, old: str, new: str, *, replace_all: bool = False) -> str:
        require_write(self.policy)
        target = self.root / path
        text = target.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError(f"String not found in {path}")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")
        return f"Edited: {path}"

    def grep(self, pattern: str, *, path: str = ".", include: str | None = None) -> str:
        args = ["grep", "-r", "-n", pattern, str(self.root / path)]
        if include:
            args += ["--include", include]
        result = subprocess.run(args, capture_output=True, text=True)
        return result.stdout or "(no matches)"

    def glob(self, pattern: str) -> str:
        matches = sorted(self.root.glob(pattern))
        return "\n".join(str(p.relative_to(self.root)) for p in matches) or "(no matches)"

    def shell(self, command: str, *, timeout: int | None = 120) -> dict[str, Any]:
        require_shell(self.policy, command)
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


