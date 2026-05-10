"""Specialist agent registry for v3 domains."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Specialist:
    name: str
    domain: str
    focus: str
    instructions: str


# ── Plugin registry ───────────────────────────────────────────────────────────
# Populated when domain packages (claudium-finance, claudium-legal, …) are imported.

_DOMAIN_POOLS: dict[str, list[Specialist]] = {}


def register_specialists(domain: str, specialists: list[Specialist]) -> None:
    """Register specialists for a domain.

    Called automatically when a domain package is imported, e.g.::

        import claudium_finance  # registers finance-audit specialists
    """
    _DOMAIN_POOLS[domain] = list(specialists)


def pool_for(domain: str) -> list[Specialist]:
    """Return the full specialist pool for a domain."""
    return list(_DOMAIN_POOLS.get(domain, []))


def select_specialists(domain: str, *, complexity: int = 1) -> list[Specialist]:
    """Select N specialists from the domain pool based on task complexity (1–3)."""
    pool = pool_for(domain)
    n = max(1, min(complexity, len(pool)))
    return pool[:n]
