"""Domain registry and inference for v3 agent teams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Domain:
    name: str
    description: str
    task_types: list[str]
    validation_criteria: list[str]
    keywords: list[str] = field(default_factory=list)
    execution_strategy: str = "parallel"  # "parallel" | "sequential"


# ── Plugin registry ───────────────────────────────────────────────────────────
# Populated when domain packages (claudium-finance, claudium-legal, …) are imported.

DOMAINS: dict[str, Domain] = {}
_FITNESS_CHECKS: dict[str, list[tuple[str, set[str]]]] = {}


def register_domain(
    domain: Domain,
    fitness_checks: list[tuple[str, set[str]]],
) -> None:
    """Register a domain and its fitness checks.

    Called automatically when a domain package is imported, e.g.::

        import claudium_finance  # registers "finance-audit"
    """
    DOMAINS[domain.name] = domain
    _FITNESS_CHECKS[domain.name] = fitness_checks


# ── Fitness scoring ───────────────────────────────────────────────────────────


def score_fitness(text: str, domain: str) -> float:
    """Domain-aware fitness score — fraction of validation criteria met (0.0–1.0)."""
    checks = _FITNESS_CHECKS.get(domain)
    if not checks:
        return 0.0
    text_lower = text.lower()
    met = sum(
        1 for _, keywords in checks
        if any(kw in text_lower for kw in keywords)
    )
    return met / len(checks)


# ── Domain inference ──────────────────────────────────────────────────────────


async def infer_domain(
    samples: list[str],
    *,
    harness: Any,
    config: Any,
    system_prompt: str = "",
) -> str:
    """Infer domain from sample prompts via one LLM call. Returns domain name or 'unknown'."""
    domain_list = "\n".join(f"- {name}: {d.description}" for name, d in DOMAINS.items())
    sample_text = "\n".join(
        f"  {i + 1}. {s}" for i, s in enumerate(samples[:10])
    )
    prompt = (
        f"Classify these agent task samples into exactly one domain.\n\n"
        f"Available domains:\n{domain_list}\n\n"
        f"Samples:\n{sample_text}\n\n"
        f"Reply with ONLY the domain name exactly as listed above. "
        f"If none match, reply 'unknown'."
    )
    result = await harness.run(
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
    )
    inferred = result.text.strip().lower()
    return inferred if inferred in DOMAINS else "unknown"
