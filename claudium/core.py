"""Claudium core agent and session API."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import TypeAdapter

from claudium.config import load_config
from claudium.harness.anthropic import AnthropicHarness
from claudium.harness.base import HarnessProtocol
from claudium.sandbox.base import SandboxPolicy
from claudium.sandbox.virtual import VirtualSandbox
from claudium.skills import (
    load_project_instructions,
    load_roles,
    load_skills,
    render_skill_prompt,
)
from claudium.types import ClaudiumConfig, ClaudiumEvent, HarnessResult, Role


async def init(
    *,
    model: str | None = None,
    sandbox: str | None = None,
    skills_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    env: dict[str, str] | None = None,
    allow_write: bool = False,
    allow_shell: bool = False,
    allowed_commands: tuple[str, ...] | list[str] | None = None,
    allow_compound_commands: bool | None = None,
    harness: HarnessProtocol | None = None,
) -> ClaudiumAgent:
    config = load_config(config_path or "claudium.toml")
    if model is not None:
        config.model = model
    if sandbox is not None:
        config.sandbox = sandbox
    if skills_dir is not None:
        path = Path(skills_dir).expanduser()
        config.skills_dir = path if path.is_absolute() else config.root / path
    if env:
        config.env.update({str(k): str(v) for k, v in env.items()})
    if allowed_commands is not None:
        config.allowed_commands = tuple(str(c) for c in allowed_commands)
    if allow_compound_commands is not None:
        config.allow_compound_commands = allow_compound_commands
    return ClaudiumAgent(
        config=config,
        sandbox_policy=SandboxPolicy(
            allow_write=allow_write,
            allow_shell=allow_shell,
            allowed_commands=config.allowed_commands,
            allow_compound_commands=config.allow_compound_commands,
        ),
        harness=harness,
    )


class ClaudiumAgent:
    def __init__(
        self,
        *,
        config: ClaudiumConfig,
        sandbox_policy: SandboxPolicy | None = None,
        harness: HarnessProtocol | None = None,
    ):
        self.config = config
        self.harness: HarnessProtocol = harness or AnthropicHarness()
        self.instructions = load_project_instructions(config.root)
        self.skills = load_skills(config.root, config.skills_dir)
        self.roles = load_roles(config.root, config.roles_dir)
        self.sandbox_policy = sandbox_policy or SandboxPolicy()
        self.state_dir = config.state_dir or config.root / ".claudium" / "sessions"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    async def session(
        self,
        session_id: str | None = None,
        *,
        role: str | None = None,
    ) -> ClaudiumSession:
        sid = session_id or "default"
        session = ClaudiumSession(agent=self, session_id=sid, role=role)
        await session._ensure_store()
        return session


class ClaudiumSession:
    def __init__(self, *, agent: ClaudiumAgent, session_id: str, role: str | None = None):
        self.agent = agent
        self.session_id = session_id
        self.session_role = role
        self.db_path = agent.state_dir / f"{session_id}.db"
        self.sandbox = VirtualSandbox(agent.config.root, agent.sandbox_policy)

    async def _ensure_store(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "create table if not exists messages "
                "(id integer primary key autoincrement, role text, content text)"
            )
            await db.commit()

    async def prompt(
        self,
        text: str,
        *,
        model: str | None = None,
        role: str | None = None,
        result: Any = None,
        secrets: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> HarnessResult | Any:
        built = self._build_prompt(text, result=result, role=role)
        history = await self._history_prompt(built)
        result_tool = _result_tool(result) if result is not None else None
        all_tools = list(tools or []) + self._mcp_tools()

        with self._grant_secrets(secrets):
            output = await self.agent.harness.run(
                prompt=history,
                system_prompt=self.agent.instructions,
                config=self._effective_config(model, role),
                result_tool=result_tool,
                tools=all_tools or None,
            )

        await self._append("user", text)
        await self._append("assistant", output.text)

        if result is not None:
            return await self._parse_with_retry(
                output, result, original_prompt=history, model=model, role=role
            )
        return output

    async def skill(
        self,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        result: Any = None,
        model: str | None = None,
        role: str | None = None,
        secrets: list[str] | None = None,
    ) -> HarnessResult | Any:
        skill = self.agent.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self.agent.skills)) or "(none)"
            raise KeyError(f"Unknown skill '{name}'. Available: {available}")
        text = render_skill_prompt(skill, args)
        return await self.prompt(text, model=model, role=role, result=result, secrets=secrets)

    async def stream(
        self,
        text: str,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> AsyncIterator[ClaudiumEvent]:
        built = self._build_prompt(text, role=role)
        history = await self._history_prompt(built)
        async for event in self.agent.harness.stream(
            prompt=history,
            system_prompt=self.agent.instructions,
            config=self._effective_config(model, role),
        ):
            yield event

    async def shell(self, command: str, *, timeout: int | None = 120) -> dict[str, Any]:
        return self.sandbox.shell(command, timeout=timeout)

    def _mcp_tools(self) -> list[dict[str, Any]]:
        servers = self.agent.config.mcp_servers
        if not servers:
            return []
        try:
            from claudium.mcp import MCPClient, tools_to_anthropic
            all_tools: list[dict[str, Any]] = []
            for server_cmd in servers:
                cmd = server_cmd if isinstance(server_cmd, list) else server_cmd.split()
                client = MCPClient(cmd)
                try:
                    client.start()
                    all_tools.extend(tools_to_anthropic(client.list_tools()))
                finally:
                    client.stop()
            return all_tools
        except Exception:
            return []

    async def task(
        self,
        task_id: str | None = None,
        *,
        role: str | None = None,
    ) -> ClaudiumTask:
        """Create a child task — isolated history, shared sandbox."""
        tid = task_id or str(uuid.uuid4())
        task = ClaudiumTask(session=self, task_id=tid, role=role)
        await task._ensure_store()
        return task

    async def _append(self, role: str, content: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "insert into messages(role, content) values (?, ?)", (role, content)
            )
            await db.commit()

    async def _messages(self) -> list[tuple[str, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "select role, content from messages order by id desc limit 12"
            )
            rows = await cursor.fetchall()
        return list(reversed([(str(r), str(c)) for r, c in rows]))

    async def _history_prompt(self, prompt: str) -> str:
        rows = await self._messages()
        if not rows:
            return prompt
        history = "\n\n".join(f"{role}: {content}" for role, content in rows)
        return f"Conversation so far:\n{history}\n\nNext:\n{prompt}"

    def _build_prompt(self, text: str, *, result: Any = None, role: str | None = None) -> str:
        parts = ["You are running inside Claudium, a headless Python agent harness."]
        selected = self._effective_role(role)
        if selected:
            parts.append(f"Role: {selected.name}\n{selected.instructions}")
        parts.append(text.strip())
        return "\n\n".join(parts)

    def _effective_config(self, model: str | None, role: str | None) -> ClaudiumConfig:
        selected = self._effective_role(role)
        resolved_model = model or (selected.model if selected else None)
        if resolved_model:
            from dataclasses import replace as dc_replace
            return dc_replace(self.agent.config, model=resolved_model)
        return self.agent.config

    def _effective_role(self, role: str | None) -> Role | None:
        name = role or self.session_role
        if not name:
            return None
        selected = self.agent.roles.get(name)
        if selected is None:
            available = ", ".join(sorted(self.agent.roles)) or "(none)"
            raise KeyError(f"Unknown role '{name}'. Available: {available}")
        return selected

    @contextmanager
    def _grant_secrets(self, names: list[str] | None):
        if not names:
            yield
            return
        missing = [n for n in names if n not in self.agent.config.env]
        if missing:
            raise KeyError(f"Unknown secret(s): {', '.join(missing)}")
        env = getattr(self.sandbox, "env", None)
        if not isinstance(env, dict):
            yield
            return
        previous = dict(env)
        try:
            env.update({n: self.agent.config.env[n] for n in names})
            yield
        finally:
            env.clear()
            env.update(previous)

    async def _parse_with_retry(
        self,
        output: HarnessResult,
        result: Any,
        *,
        original_prompt: str,
        model: str | None,
        role: str | None,
        retries: int | None = None,
    ) -> Any:
        max_retries = retries if retries is not None else self.agent.config.typed_retries
        last_error: Exception | None = None
        current = output
        for attempt in range(max_retries + 1):
            try:
                return _parse_typed_result(current.text, result)
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                schema = TypeAdapter(result).json_schema()
                repair_prompt = (
                    f"{original_prompt}\n\n"
                    "Structured output validation failed.\n"
                    f"Error: {exc}\n\n"
                    "Return valid JSON satisfying this schema:\n"
                    f"{json.dumps(schema, indent=2, sort_keys=True)}"
                )
                current = await self.agent.harness.run(
                    prompt=repair_prompt,
                    system_prompt=self.agent.instructions,
                    config=self._effective_config(model, role),
                    result_tool=_result_tool(result),
                )
        msg = f"Structured output failed after {max_retries} retries: {last_error}"
        raise ValueError(msg) from last_error


class ClaudiumTask:
    """Child task — isolated conversation history, shared sandbox with parent session."""

    def __init__(self, *, session: ClaudiumSession, task_id: str, role: str | None = None) -> None:
        self.agent = session.agent
        self.task_id = task_id
        self.session_role = role
        self.sandbox = session.sandbox
        self.db_path = session.agent.state_dir / f"task-{task_id}.db"

    async def _ensure_store(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "create table if not exists messages "
                "(id integer primary key autoincrement, role text, content text)"
            )
            await db.commit()

    async def prompt(
        self, text: str, *, model: str | None = None, result: Any = None
    ) -> HarnessResult | Any:
        parts = ["You are running inside Claudium as a focused child task."]
        role_obj = self._effective_role()
        if role_obj:
            parts.append(f"Role: {role_obj.name}\n{role_obj.instructions}")
        parts.append(text.strip())
        built = "\n\n".join(parts)
        result_tool = _result_tool(result) if result is not None else None
        config = self.agent.config
        if model or (role_obj and role_obj.model):
            from dataclasses import replace as dc_replace
            config = dc_replace(config, model=model or role_obj.model)  # type: ignore[arg-type]
        output = await self.agent.harness.run(
            prompt=built,
            system_prompt=self.agent.instructions,
            config=config,
            result_tool=result_tool,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("insert into messages(role, content) values (?, ?)", ("user", text))
            await db.execute(
                "insert into messages(role, content) values (?, ?)", ("assistant", output.text)
            )
            await db.commit()
        if result is not None:
            raw = output.text.strip()
            value: Any = json.loads(raw) if raw.startswith(("{", "[")) else raw
            return TypeAdapter(result).validate_python(value)
        return output

    async def skill(
        self, name: str, *, args: dict[str, Any] | None = None,
        result: Any = None, model: str | None = None,
    ) -> HarnessResult | Any:
        skill = self.agent.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self.agent.skills)) or "(none)"
            raise KeyError(f"Unknown skill '{name}'. Available: {available}")
        return await self.prompt(render_skill_prompt(skill, args), model=model, result=result)

    def _effective_role(self) -> Role | None:
        if not self.session_role:
            return None
        role = self.agent.roles.get(self.session_role)
        if role is None:
            available = ", ".join(sorted(self.agent.roles)) or "(none)"
            raise KeyError(f"Unknown role '{self.session_role}'. Available: {available}")
        return role


def _result_tool(result: Any) -> dict[str, Any]:
    schema = TypeAdapter(result).json_schema()
    return {
        "name": "structured_result",
        "description": "Return the final structured result.",
        "input_schema": schema,
    }


def _parse_typed_result(text: str, result: Any) -> Any:
    raw = text.strip()
    value: Any = raw
    if raw.startswith("{") or raw.startswith("["):
        value = json.loads(raw)
    return TypeAdapter(result).validate_python(value)
