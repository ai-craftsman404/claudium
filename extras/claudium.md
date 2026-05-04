---
description: Build and run Claudium agents — the Anthropic-native agent harness for Python
---

You are helping the user work with Claudium, an Anthropic-native agent harness framework for Python.

## What Claudium provides

- **Sessions** — stateful, SQLite-backed conversation history that resumes across runs
- **Skills** — reusable prompt templates defined as Markdown with YAML frontmatter
- **Roles** — persona definitions that override the model and inject system instructions
- **Tasks** — isolated child conversations that share a parent session's sandbox
- **Sandbox** — policy-gated filesystem access (read / write / shell), explicit opt-in
- **Typed outputs** — structured results via Anthropic's native tool-use (Pydantic models)
- **Streaming** — async event stream from the model
- **MCP** — first-class MCP server passthrough and Claudium-as-MCP-server

## Installation

```bash
pip install claudium                  # core
pip install claudium[server]          # + FastAPI webhook server
pip install claudium[mcp]             # + MCP server support
pip install claudium[server,mcp]      # everything
```

## Scaffold a project

```bash
claudium init my-project
cd my-project
claudium run --prompt "Hello, agent"
```

Generates: `claudium.toml`, `CLAUDE.md`, `agents/default.py`, `.agents/skills/default.md`

## Core API

```python
import asyncio
from claudium import init

async def main():
    agent = await init()                       # reads claudium.toml
    session = await agent.session("s1")        # persistent — resumes on re-open
    result = await session.prompt("Summarise this PR")
    print(result.text)

asyncio.run(main())
```

## Typed output (Pydantic)

```python
from pydantic import BaseModel

class Triage(BaseModel):
    severity: str
    labels: list[str]

result = await session.prompt("Triage this issue", result=Triage)
print(result.severity)   # e.g. "high"
```

## Skills

Define `.agents/skills/triage.md`:

```markdown
---
name: triage
description: Classify a GitHub issue by severity and labels
---
Analyse issue #{{ issue_number }} carefully and return its severity and relevant labels.
```

Invoke:

```python
result = await session.skill("triage", args={"issue_number": 42})
```

## Roles

Define `.agents/roles/analyst.md`:

```markdown
---
name: analyst
model: claude-opus-4-5
---
You are a precise data analyst. Always cite sources and show your reasoning.
```

Use:

```python
session = await agent.session("s1", role="analyst")
```

## Tasks (isolated child conversations)

```python
task = await session.task("review-pr-42")
result = await task.prompt("Review the changes in this diff: ...")
# task has its own history; shares the session's sandbox
```

## Streaming

```python
async for event in session.stream("Explain this code"):
    if event.type == "text_delta":
        print(event.data["text"], end="", flush=True)
```

## Sandbox

```python
from claudium.types import ClaudiumConfig
from claudium.sandbox.base import SandboxPolicy
from claudium.sandbox.virtual import VirtualSandbox

policy = SandboxPolicy(allow_write=True, allow_shell=False)
sandbox = VirtualSandbox(root=Path("./workspace"), policy=policy)
```

## Configuration (claudium.toml)

```toml
[agent]
model = "claude-opus-4-5"
sandbox = "virtual"

[sandbox]
allow_write = true
allow_shell = false
allowed_commands = []

[mcp]
servers = []
```

## CLI reference

| Command | Description |
|---|---|
| `claudium init <name>` | Scaffold a new project |
| `claudium run --prompt "..."` | Run a prompt against the default session |
| `claudium run --skill <name>` | Invoke a named skill |
| `claudium run --stream` | Stream the response to stdout |
| `claudium run --session <id>` | Use a named session |
| `claudium dev [--port 2024]` | Start the local webhook dev server |
| `claudium build --target <t>` | Generate deployment files |
| `claudium mcp` | Start the Claudium MCP server (stdio) |

## Deployment targets

```bash
claudium build --target docker    # Dockerfile + .dockerignore
claudium build --target railway   # railway.json
claudium build --target fly       # fly.toml
claudium build --target render    # render.yaml
claudium build --target ci        # .github/workflows/ci.yml
```

## MCP server (Claude Code integration)

Add to your Claude Code `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "claudium": {
      "command": "claudium",
      "args": ["mcp"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

Exposed MCP tools: `claudium_prompt`, `claudium_skill`, `claudium_list_skills`

## Harness injection (testing without API key)

```python
from claudium.core import ClaudiumAgent
from claudium.types import ClaudiumConfig, HarnessResult

class MockHarness:
    async def run(self, *, prompt, system_prompt, config, result_tool=None, tools=None):
        return HarnessResult(text="mock response")
    async def stream(self, **_):
        ...  # yield ClaudiumEvent items

agent = ClaudiumAgent(config=ClaudiumConfig(root=Path(".")), harness=MockHarness())
```

When helping the user, prefer skills over inline prompts for reusable tasks. Use tasks when parallel or isolated sub-conversations are needed. Always suggest typed outputs (`result=MyModel`) when structured data is expected.
