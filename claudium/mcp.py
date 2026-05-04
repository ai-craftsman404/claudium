"""MCP (Model Context Protocol) tool integration."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class MCPClient:
    """Minimal stdio-based MCP client."""

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._proc: subprocess.Popen | None = None
        self._seq = 0

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._initialize()

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None

    def list_tools(self) -> list[dict[str, Any]]:
        response = self._request("tools/list", {})
        return response.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        response = self._request("tools/call", {"name": name, "arguments": arguments})
        content = response.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]
        return response

    def _initialize(self) -> None:
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "claudium", "version": "0.1.0"},
        })

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("MCP client not started")
        self._seq += 1
        msg = json.dumps({"jsonrpc": "2.0", "id": self._seq, "method": method, "params": params})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})


def tools_to_anthropic(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool-use format."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {"type": "object", "properties": {}}),
        }
        for t in mcp_tools
    ]
