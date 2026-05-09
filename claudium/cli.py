"""Claudium CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="claudium", help="The agent harness framework for Claude.")
audit_app = typer.Typer(name="audit", help="Compliance audit log commands.")
app.add_typer(audit_app, name="audit")
console = Console()


@app.command()
def init(name: str = typer.Argument(..., help="Project directory name")) -> None:
    """Initialise a new Claudium project."""
    root = Path(name)
    if root.exists():
        console.print(f"[red]Directory already exists:[/red] {name}")
        raise typer.Exit(1)

    root.mkdir(parents=True)
    (root / ".agents" / "skills").mkdir(parents=True)
    (root / ".agents" / "roles").mkdir(parents=True)
    (root / "agents").mkdir(parents=True)

    (root / "claudium.toml").write_text(
        '[agent]\nmodel = "claude-opus-4-5"\nsandbox = "virtual"\n\n'
        "[sandbox]\nallow_write = false\nallow_shell = false\nallowed_commands = []\n\n"
        "[mcp]\nservers = []\n",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text(
        "# Project Instructions\n\nDescribe your project context here.\n",
        encoding="utf-8",
    )
    (root / ".agents" / "skills" / "default.md").write_text(
        "---\nname: default\ndescription: Default skill\n---\n\nYou are a helpful agent.\n",
        encoding="utf-8",
    )
    (root / "agents" / "default.py").write_text(
        'triggers = {"webhook": True}\n\n\nasync def default(context):\n'
        "    agent = await context.init()\n"
        "    session = await agent.session(context.agent_id)\n"
        "    result = await session.prompt(context.payload[\"prompt\"])\n"
        "    return {\"text\": result.text}\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(".claudium/\n__pycache__/\n.env\n", encoding="utf-8")

    console.print(f"[green]Created project:[/green] {name}")
    console.print(f"  cd {name}")
    console.print("  claudium run --prompt 'Hello'")


@app.command()
def run(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to run"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
    skill: str = typer.Option(None, "--skill", "-s", help="Skill name to invoke"),
    stream: bool = typer.Option(False, "--stream", help="Stream output"),
    session_id: str = typer.Option("default", "--session", help="Session ID"),
) -> None:
    """Run a prompt or skill against a Claudium agent."""
    asyncio.run(_run(prompt=prompt, model=model, skill=skill, stream=stream, session_id=session_id))


async def _run(
    *,
    prompt: str,
    model: str | None,
    skill: str | None,
    stream: bool,
    session_id: str,
) -> None:
    from claudium import init as claudium_init

    agent = await claudium_init(model=model)
    session = await agent.session(session_id)

    if stream:
        async for event in session.stream(prompt):
            if event.type == "text_delta":
                console.print(event.data.get("text", ""), end="")
        console.print()
        return

    if skill:
        result = await session.skill(skill, args={"prompt": prompt})
    else:
        result = await session.prompt(prompt)

    console.print(result.text if hasattr(result, "text") else result)


@app.command()
def dev(
    port: int = typer.Option(2024, "--port", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes"),
) -> None:
    """Start the local webhook development server."""
    try:
        import uvicorn

        from claudium.server import create_app
        console.print(f"[green]Claudium dev server[/green] → http://{host}:{port}")
        uvicorn.run(create_app(), host=host, port=port, reload=reload)
    except ImportError:
        console.print("[red]Install claudium[server] to use the dev server.[/red]")
        raise typer.Exit(1)


@app.command()
def build(
    target: str = typer.Option(..., "--target", "-t", help="docker | railway | fly | render | ci"),
) -> None:
    """Generate deployment files for a target platform."""
    from claudium.build import build as do_build
    try:
        files = do_build(target, Path.cwd())
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    for path, content in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            console.print(f"[yellow]skipped (exists):[/yellow] {path.relative_to(Path.cwd())}")
            continue
        path.write_text(content, encoding="utf-8")
        console.print(f"[green]created:[/green] {path.relative_to(Path.cwd())}")


@app.command()
def mcp(
    root: Path = typer.Option(None, "--root", help="Project root (default: cwd)"),
) -> None:
    """Start the Claudium MCP server (stdio transport for Claude Code)."""
    try:
        from claudium.mcp_server import serve
        serve(root)
    except ImportError:
        console.print("[red]Install claudium[mcp] to use the MCP server.[/red]")
        raise typer.Exit(1)


@app.command()
def trace(
    session: str = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
) -> None:
    """Show recent harness call traces (latency, tokens, model)."""
    asyncio.run(_trace(session=session, limit=limit))


async def _trace(*, session: str | None, limit: int) -> None:
    import aiosqlite
    from rich.table import Table

    from claudium import init as claudium_init

    agent = await claudium_init()
    dbs = sorted(agent.state_dir.glob("*.db"))
    table = Table("Session", "Skill", "Model", "ms", "In", "Out", "Time")
    for db_path in dbs:
        async with aiosqlite.connect(db_path) as db:
            try:
                where = f"WHERE session_id = '{session}'" if session else ""
                cursor = await db.execute(
                    f"SELECT session_id, skill, model, latency_ms, input_tokens,"
                    f" output_tokens, created_at FROM call_log {where}"
                    f" ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                for row in await cursor.fetchall():
                    table.add_row(*[str(v) if v is not None else "-" for v in row])
            except Exception:
                pass
    console.print(table)


@app.command()
def calibrate(
    skill: str = typer.Argument(..., help="Skill name to calibrate"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Dataset file (one prompt per line)"),
    team_size: int = typer.Option(3, "--team-size", help="Number of sub-agents"),
    window: int = typer.Option(10, "--window", help="Rolling average window size"),
) -> None:
    """Calibrate routing weights for a skill against a sample dataset."""
    asyncio.run(_calibrate(skill=skill, dataset=dataset, team_size=team_size, window=window))


async def _calibrate(*, skill: str, dataset: Path, team_size: int, window: int) -> None:
    from rich.table import Table

    from claudium import init as claudium_init

    if not dataset.exists():
        console.print(f"[red]Dataset file not found:[/red] {dataset}")
        raise typer.Exit(1)

    raw = dataset.read_text(encoding="utf-8").splitlines()
    samples = [line.strip() for line in raw if line.strip()]
    if not samples:
        console.print("[red]Dataset is empty.[/red]")
        raise typer.Exit(1)

    agent = await claudium_init()
    orch = await agent.orchestrator("calibrate", weight_window=window)
    await orch.team(team_size)

    n = len(samples)
    console.print(
        f"[green]Calibrating[/green] skill=[bold]{skill}[/bold] samples={n} team={team_size}"
    )
    cal = await orch.calibrate(skill, samples)

    table = Table("Agent", "Weight", "Runs")
    for w in cal.weights:
        table.add_row(str(w.agent_index), f"{w.weight:.3f}", str(w.run_count))
    console.print(table)
    console.print(f"Mean agreement: [bold]{cal.mean_agreement:.2f}[/bold]")


@audit_app.command("export")
def audit_export(
    session: str = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    since: str = typer.Option(None, "--since", help="ISO date lower bound (e.g. 2026-05-01)"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json or csv"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
) -> None:
    """Export audit log as JSON or CSV for compliance and regulatory reporting."""
    asyncio.run(_audit_export(session=session, since=since, fmt=fmt, output=output))


async def _audit_export(
    *,
    session: str | None,
    since: str | None,
    fmt: str,
    output: Path | None,
) -> None:
    from claudium import init as claudium_init
    from claudium.audit import export_audit

    if fmt not in ("json", "csv"):
        console.print("[red]--format must be 'json' or 'csv'[/red]")
        raise typer.Exit(1)

    agent = await claudium_init()
    db_paths = sorted(agent.state_dir.glob("*.db"))
    report = await export_audit(db_paths, session=session, since=since, fmt=fmt)

    if output:
        output.write_text(report, encoding="utf-8")
        console.print(f"[green]Audit report written to:[/green] {output}")
    else:
        console.print(report, highlight=False)


if __name__ == "__main__":
    app()
