"""Sandbox layer — backward-compatible imports."""

from claudium.sandbox.base import SandboxFileInfo, SandboxPolicy
from claudium.sandbox.virtual import VirtualSandbox

__all__ = ["SandboxFileInfo", "SandboxPolicy", "VirtualSandbox"]
