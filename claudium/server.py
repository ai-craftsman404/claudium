"""FastAPI webhook server for Claudium agents."""

from __future__ import annotations

import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


class AgentContext:
    """Runtime context passed to each webhook agent handler."""

    def __init__(self, *, agent_id: str, payload: dict[str, Any], root: Path) -> None:
        self.agent_id = agent_id
        self.payload = payload
        self._root = root

    async def init(self, **kwargs: Any):
        from claudium.core import init
        return await init(config_path=self._root / "claudium.toml", **kwargs)


def _load_agent_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _discover_agents(agents_dir: Path) -> dict[str, Any]:
    if not agents_dir.exists():
        return {}
    handlers: dict[str, Any] = {}
    for path in sorted(agents_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _load_agent_module(path)
            triggers = getattr(module, "triggers", {})
            if not triggers.get("webhook"):
                continue
            handler = getattr(module, path.stem, None)
            if callable(handler):
                handlers[path.stem] = handler
        except Exception:
            pass
    return handlers


def create_app(root: Path | None = None) -> FastAPI:
    root = (root or Path.cwd()).resolve()
    agents_dir = root / "agents"
    _handlers: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _handlers.update(_discover_agents(agents_dir))
        yield
        _handlers.clear()

    app = FastAPI(title="Claudium", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agents": ", ".join(sorted(_handlers))}

    @app.post("/agents/{name}/{agent_id}")
    async def handle_agent(name: str, agent_id: str, request: Request) -> JSONResponse:
        handler = _handlers.get(name)
        if handler is None:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not found or not a webhook agent")
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            body = {}
        payload = body.get("payload", body)
        context = AgentContext(agent_id=agent_id, payload=payload, root=root)
        try:
            result = await handler(context)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(result if isinstance(result, dict) else {"result": result})

    return app
