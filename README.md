<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/github/actions/workflow/status/ai-craftsman404/claudium/ci.yml?style=flat-square&label=CI&logo=github" alt="CI"/>
  <img src="https://img.shields.io/badge/built%20for-Claude-blueviolet?style=flat-square" alt="Built for Claude"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/status-active%20development-orange?style=flat-square" alt="Active Development"/>
</p>

<h1 align="center">Claudium</h1>

<p align="center"><strong>Ship Claude-powered agents in minutes, not days.</strong></p>

<p align="center">
The Anthropic-native agent harness for Python — giving you skills, sessions,<br/>
sandboxing, typed outputs, streaming, and one-command deployment.<br/>
All the infrastructure you need. None you don't.
</p>

---

## Why Claudium?

Building production agents with Claude involves the same boilerplate every time — session history, prompt engineering for structured outputs, safe file and shell access, deployment. Claudium handles all of it so you focus on what your agent actually does.

```python
from pydantic import BaseModel
from claudium import init

class TriageResult(BaseModel):
    severity: str
    labels: list[str]
    summary: str

async def main():
    agent = await init(model="claude-opus-4-5")
    session = await agent.session("triage-42")
    result = await session.skill("triage", args={"issue": 42}, result=TriageResult)
    print(result.severity)  # "critical"
```

That's a fully stateful, typed, Claude-powered triage agent. No prompt engineering for JSON. No session management. No boilerplate.

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
pip install "claudium[server]"     # webhook server — FastAPI + SSE
pip install "claudium[sandboxes]"  # remote sandboxes — E2B and more
```

---

## Quick Start

```bash
claudium init my-agent
cd my-agent
claudium run --prompt "Triage issue #42"
```

Three commands. Running agent.

---

## What You Get

| | Capability | What it means |
|---|---|---|
| 📝 | **Markdown skills** | Define reusable agent workflows in `.agents/skills/*.md` — no code required |
| 🔒 | **Secure sandbox** | Filesystem and shell access behind explicit opt-in policy — locked down by default |
| 💾 | **Stateful sessions** | Resume agent context across runs with stable session IDs, SQLite-backed |
| 🧩 | **Child tasks** | Spawn focused sub-agents with isolated history and shared sandbox |
| 🎯 | **Typed outputs** | Return validated Pydantic models via Claude's native tool-use — no delimiter hacks |
| ⚡ | **Prompt caching** | System prompts and skill instructions cached automatically — lower cost, faster responses |
| 🌊 | **Streaming** | First-class streaming via `session.stream()`, CLI, or SSE webhook |
| 🔌 | **MCP integration** | Pass MCP server tools directly into Claude's tool-use layer |
| 🪝 | **Webhook agents** | Expose any agent as `POST /agents/{name}/{agent_id}` with one decorator |
| 🚀 | **One-command deploy** | Generate Docker, Railway, Fly.io, and Render configs with `claudium build` |

---

## Skills — Reusable Agent Workflows

Skills are Markdown files. Write once, invoke anywhere.

`.agents/skills/triage.md`:

```markdown
---
name: triage
description: Classify a GitHub issue by severity, labels, and assignee
---

You are an expert issue triage agent.

Given an issue number and repository:
1. Classify severity — critical, high, medium, or low
2. Suggest appropriate labels
3. Write a one-sentence summary
4. Suggest an assignee if determinable
```

Invoke it with full type safety:

```python
result = await session.skill(
    "triage",
    args={"issue_number": 42, "repo": "org/repo"},
    result=TriageResult,
)
```

---

## Roles — Smart Model Routing

Route tasks to the right Claude model automatically.

`.agents/roles/analyst.md`:

```markdown
---
name: analyst
model: claude-sonnet-4-5
---

You are a senior technical analyst. Be concise and evidence-based.
```

Assign Haiku to fast tasks, Sonnet to standard work, Opus to deep reasoning — all in configuration, zero code changes.

---

## Sessions — Stateful by Default

Every session persists conversation history automatically.

```python
# Day 1
session = await agent.session("customer-42")
await session.prompt("The user reports a login failure on mobile.")

# Day 2 — same session, full context restored
session = await agent.session("customer-42")
await session.prompt("Any update on that login issue?")  # Claude remembers
```

---

## Child Tasks — Focused Sub-Agents

Spawn isolated tasks that share the parent sandbox.

```python
session = await agent.session("pipeline-run-1")
summary_task = await session.task("summarise", role="analyst")
result = await summary_task.skill("summarise", args={"doc": "report.pdf"})
```

Isolated history. Shared filesystem. Clean separation.

---

## Security Model

Claudium starts fully locked down and requires explicit opt-in for every capability.

```python
agent = await init(
    model="claude-opus-4-5",
    allow_write=True,           # disabled by default
    allow_shell=True,           # disabled by default
    allowed_commands=["git"],   # empty = nothing allowed
)
```

- Writes disabled until `allow_write=True`
- Shell disabled until `allow_shell=True`
- Compound shell syntax (`&&`, `|`, `;`) blocked by default
- Secrets never injected into prompts — granted per-call only via `secrets=[...]`

---

## Webhook Agents

Turn any agent into a live HTTP endpoint.

`agents/triage.py`:

```python
triggers = {"webhook": True}

async def triage(context):
    agent = await context.init()
    session = await agent.session(context.agent_id)
    result = await session.skill("triage", args=context.payload, result=TriageResult)
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

## Streaming

```bash
claudium run --stream --prompt "Analyse this dataset"
```

```python
async for event in session.stream("Analyse this dataset"):
    if event.type == "text_delta":
        print(event.data["text"], end="", flush=True)
```

---

## MCP Integration

Connect MCP servers in `claudium.toml` and their tools are available to Claude automatically.

```toml
[mcp]
servers = ["npx -y @modelcontextprotocol/server-filesystem ."]
```

No adapter. No boilerplate. Claude calls MCP tools natively.

---

## Project Layout

```
my-agent/
├── CLAUDE.md              ← agent system prompt
├── claudium.toml          ← project config
├── .agents/
│   ├── skills/
│   │   └── triage.md      ← reusable skill definitions
│   └── roles/
│       └── analyst.md     ← model + behaviour scoping
├── agents/
│   └── triage.py          ← webhook agent
└── tests/
```

---

## Configuration

`claudium.toml`:

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

---

## Deployment

Generate production-ready deployment files in one command:

```bash
claudium build --target docker    # Dockerfile + .dockerignore
claudium build --target railway   # railway.toml
claudium build --target fly       # fly.toml
claudium build --target render    # render.yaml
claudium build --target ci        # GitHub Actions workflow
```

---

## Use Cases

Claudium is built for agents that do real work — not chat interfaces. Here are three production scenarios out of the box.

---

### 1. GitHub Issue Triage

Automatically classify, label, and route incoming issues — triggered by webhook, returns a typed result.

```console
$ claudium run --skill triage --prompt "Issue #142: Login fails on Safari macOS 14"

  Severity  →  high
  Labels    →  bug · browser-compat · auth
  Assignee  →  @auth-team
  Summary   →  Safari SameSite cookie regression affecting macOS 14+
```

```python
from pydantic import BaseModel
from claudium import init

class TriageResult(BaseModel):
    severity: str
    labels: list[str]
    assignee: str
    summary: str

agent  = await init()
session = await agent.session("github-triage")

result = await session.skill(
    "triage",
    args={"issue_number": 142, "title": "Login fails on Safari macOS 14"},
    result=TriageResult,
)
# TriageResult(
#   severity = "high",
#   labels   = ["bug", "browser-compat", "auth"],
#   assignee = "@auth-team",
#   summary  = "Safari SameSite cookie regression affecting macOS 14+"
# )
```

---

### 2. Automated PR Code Review

Spawn parallel child tasks — each reviewer has isolated context but shares the same sandbox. No cross-contamination, no boilerplate.

```console
$ claudium run --skill pr-review --prompt "PR #99: OAuth2 + new DB access layer"

  [security]     2 findings — SQL injection risk db.py:142 · missing CSRF token
  [performance]  1 finding  — N+1 query in user_loader(), suggest eager load
  [style]        Passed
```

```python
session = await agent.session("pr-review-99")

security    = await session.task("security",    role="security-analyst")
performance = await session.task("performance", role="perf-analyst")

sec_result  = await security.prompt("Review auth.py for vulnerabilities")
perf_result = await performance.prompt("Review db.py for N+1 query patterns")

# Two focused reviewers — isolated history, shared filesystem, typed findings
```

---

### 3. Persistent Customer Support

Sessions resume full conversation history across every interaction. Route to the right tier automatically via roles — no extra plumbing.

```console
$ claudium run --session customer-7821 --prompt "Still having login issues since Monday"

  Context loaded  →  4 prior messages (login failure first reported 3 days ago)
  Tier            →  support-tier1 → escalating to support-tier2
  Response        →  I can see you've been dealing with this since Monday.
                     Let's reset your session tokens — here's exactly how...
```

```python
session  = await agent.session(f"customer-{customer_id}", role="support-tier1")
response = await session.prompt(customer_message)

# Claude automatically recalls the full prior conversation.
# Role assigns the right model and persona — no code changes needed to escalate.
```

---

## Development

```bash
git clone https://github.com/ai-craftsman404/claudium
cd claudium
pip install -e ".[dev,server]"
pytest tests/ -v
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting large changes.

---

## License

MIT — see [LICENSE](LICENSE) for details.
