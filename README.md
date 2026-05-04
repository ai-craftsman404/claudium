# Claudium

> The agent harness framework for Claude.

Claudium gives you Markdown skills, stateful sessions, sandboxed filesystem and
shell access, typed Pydantic outputs via Claude's native tool-use, streaming
events, file-based webhook routes, and deployment-ready project structure —
all built directly on the Anthropic SDK.

Use it to build issue triage agents, customer support agents, document
processing pipelines, data analysis agents, and workflow agents that need
controlled access to files, commands, tools, and structured outputs.

> **Active Development**
> Claudium is under active development. Pin your dependencies and review
> changelogs before updating.

---

## Install

```bash
pip install claudium
```

```bash
uv add claudium
```

Optional extras:

```bash
pip install "claudium[sandboxes]"   # remote sandbox providers (E2B, Daytona, Modal)
pip install "claudium[server]"      # webhook server (FastAPI + SSE)
```

---

## Quick Start

```bash
claudium init my-agent
cd my-agent
claudium run --prompt "Triage issue #42"
```

---

## Python API

```python
from pydantic import BaseModel
from claudium import init


class TriageResult(BaseModel):
    severity: str                  # "critical" | "high" | "medium" | "low"
    labels: list[str]
    summary: str
    assignee: str | None


async def main():
    agent = await init(
        model="claude-opus-4-5",
        allow_write=False,
        allow_shell=False,
    )
    session = await agent.session("triage-42")
    result = await session.skill(
        "triage",
        args={"issue_number": 42, "repo": "org/repo"},
        result=TriageResult,
    )
    print(result.severity, result.labels)
```

---

## What Claudium Gives You

| Capability         | What it means                                                         |
|--------------------|-----------------------------------------------------------------------|
| Markdown skills    | Reusable workflows in `.agents/skills/*.md`                           |
| Project context    | `CLAUDE.md` loaded as the agent system prompt automatically           |
| Roles              | Scope model and behaviour per call with `.agents/roles/*.md`          |
| Sessions           | Resume agent state with stable session IDs (SQLite-backed)            |
| Tasks              | Run focused child tasks with isolated history and shared sandbox      |
| Sandbox            | Read, write, edit, grep, glob, shell — behind explicit opt-in policy  |
| Secret grants      | Secrets never injected into prompts; mounted per-call only            |
| Typed outputs      | Pydantic results via Claude's native tool-use — no prompt hacks       |
| Prompt caching     | Baked in by default on system prompt and skill instructions           |
| Streaming          | `session.stream(...)`, `claudium run --stream`, or SSE                |
| MCP               | First-class MCP server passthrough for Claude tool-use                |
| Webhooks           | Expose `agents/*.py` as `POST /agents/{name}/{agent_id}`              |
| Deployment         | Docker, Railway, Fly.io, Render, Vercel starter files                 |

---

## Project Layout

```
CLAUDE.md
claudium.toml
.agents/
  roles/
    analyst.md
  skills/
    triage.md
agents/
  default.py
```

---

## Skill File

`.agents/skills/triage.md`:

```markdown
---
name: triage
description: Analyse and classify a GitHub issue by severity, labels, and suggested assignee
---

You are an expert issue triage agent.

Given an issue number and repository, retrieve the issue details and:
1. Classify severity as one of: critical, high, medium, low
2. Suggest appropriate labels
3. Write a one-sentence summary
4. Suggest an assignee if determinable from context
```

---

## Role File

`.agents/roles/analyst.md`:

```markdown
---
name: analyst
description: Senior technical analyst — concise, evidence-based responses
model: claude-sonnet-4-5
---

You are a senior technical analyst. Be concise and cite evidence.
Avoid speculation. If uncertain, say so explicitly.
```

Roles can override the model — useful for routing fast/cheap tasks to Haiku
and deep reasoning to Opus.

---

## Streaming

```bash
claudium run --stream --prompt "Triage issue #42"
```

```python
async for event in session.stream("Triage issue #42"):
    print(event.type, event.data)
```

---

## Security Model

Claudium starts locked down:

- Writes disabled until `allow_write=True`
- Shell disabled until `allow_shell=True`
- Compound shell syntax blocked by default
- `allowed_commands` tuple as an explicit allowlist
- Secrets never appear in prompts; granted per-call via `secrets=[...]`

```python
agent = await init(
    model="claude-opus-4-5",
    allow_write=True,
    allow_shell=True,
    allowed_commands=["git", "gh"],
)
```

---

## Webhook Agent

`agents/triage.py`:

```python
triggers = {"webhook": True}


async def triage(context):
    agent = await context.init()
    session = await agent.session(context.agent_id)
    result = await session.skill(
        "triage",
        args=context.payload,
        result=TriageResult,
    )
    return result.model_dump()
```

```bash
claudium dev --port 2024
```

```bash
curl http://127.0.0.1:2024/agents/triage/run-1 \
  -H "Content-Type: application/json" \
  -d '{"payload": {"issue_number": 42, "repo": "org/repo"}}'
```

---

## MCP

```toml
# claudium.toml
[mcp]
servers = ["filesystem", "github"]
```

MCP tools are passed directly to Claude's tool-use layer — no adapter needed.

---

## Deployment

```bash
claudium build --target docker
claudium build --target railway
claudium build --target fly

claudium deploy --target fly
```

---

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

---

## claudium.toml

```toml
[agent]
model = "claude-opus-4-5"
sandbox = "virtual"

[sandbox]
allow_write = false
allow_shell = false
allowed_commands = []

[mcp]
servers = []
```
