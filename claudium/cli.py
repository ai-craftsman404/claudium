"""Claudium CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="claudium", help="The agent harness framework for Claude.")
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


if __name__ == "__main__":
    app()
