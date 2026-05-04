"""Deployment file generation for claudium build --target."""

from __future__ import annotations

from pathlib import Path

_DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[server]"
EXPOSE 2024
CMD ["claudium", "dev", "--host", "0.0.0.0", "--port", "2024"]
"""

_DOCKERIGNORE = """\
.claudium/
.venv/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
"""

_RAILWAY_TOML = """\
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "claudium dev --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
"""

_FLY_TOML = """\
app = "{app_name}"
primary_region = "lhr"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 2024
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
"""

_RENDER_YAML = """\
services:
  - type: web
    name: {app_name}
    env: python
    buildCommand: pip install -e ".[server]"
    startCommand: claudium dev --host 0.0.0.0 --port $PORT
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
"""

_GITHUB_CI = """\
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest tests/ -v
"""


def build(target: str, root: Path) -> list[tuple[Path, str]]:
    """Return list of (path, content) pairs to write for the given target."""
    app_name = _app_name(root)

    if target == "docker":
        return [
            (root / "Dockerfile", _DOCKERFILE),
            (root / ".dockerignore", _DOCKERIGNORE),
        ]
    if target == "railway":
        return [
            (root / "Dockerfile", _DOCKERFILE),
            (root / ".dockerignore", _DOCKERIGNORE),
            (root / "railway.toml", _RAILWAY_TOML),
        ]
    if target == "fly":
        return [
            (root / "Dockerfile", _DOCKERFILE),
            (root / ".dockerignore", _DOCKERIGNORE),
            (root / "fly.toml", _FLY_TOML.format(app_name=app_name)),
        ]
    if target == "render":
        return [
            (root / "Dockerfile", _DOCKERFILE),
            (root / ".dockerignore", _DOCKERIGNORE),
            (root / "render.yaml", _RENDER_YAML.format(app_name=app_name)),
        ]
    if target == "ci":
        return [
            (root / ".github" / "workflows" / "ci.yml", _GITHUB_CI),
        ]
    raise ValueError(f"Unknown target '{target}'. Available: docker, railway, fly, render, ci")


def _app_name(root: Path) -> str:
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        config_path = root / "claudium.toml"
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
            name = data.get("agent", {}).get("name")
            if name:
                return str(name)
    except Exception:
        pass
    return root.name or "claudium-agent"
