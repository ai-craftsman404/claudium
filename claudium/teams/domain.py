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


# ── Legal/Compliance domain ───────────────────────────────────────────────────

LEGAL_COMPLIANCE = Domain(
    name="legal-compliance",
    description=(
        "Contract review, clause extraction, obligation validation, "
        "risk classification, regulatory compliance"
    ),
    task_types=[
        "clause-extraction",
        "obligation-validation",
        "risk-classification",
    ],
    validation_criteria=[
        "clause_type_identified",
        "risk_level_assigned",
        "obligation_extracted",
        "party_identified",
    ],
    keywords=[
        "contract", "clause", "indemnif", "liability", "termination",
        "obligation", "covenant", "warranty", "breach", "jurisdiction",
        "governing law", "gdpr", "compliance", "regulatory", "legal",
        "agreement", "nda", "sla", "arbitration", "force majeure",
    ],
    execution_strategy="parallel",
)

# ── Finance/Audit domain ──────────────────────────────────────────────────────

FINANCE_AUDIT = Domain(
    name="finance-audit",
    description=(
        "Financial transaction review, audit finding extraction, "
        "AML/SOX/BSA compliance verification, risk and anomaly detection"
    ),
    task_types=[
        "transaction-audit",
        "risk-analysis",
        "compliance-check",
    ],
    validation_criteria=[
        "transaction_identified",
        "regulation_named",
        "risk_flagged",
        "evidence_cited",
    ],
    keywords=[
        "transaction", "invoice", "audit", "sox", "aml", "bsa", "fatf",
        "journal entry", "ledger", "reconciliation", "compliance", "risk",
        "anomaly", "suspicious", "threshold", "control", "finding",
        "financial statement", "balance sheet", "revenue", "expenditure",
    ],
    execution_strategy="sequential",
)

# ── Domain registry ───────────────────────────────────────────────────────────

DOMAINS: dict[str, Domain] = {
    LEGAL_COMPLIANCE.name: LEGAL_COMPLIANCE,
    FINANCE_AUDIT.name: FINANCE_AUDIT,
}

# ── Fitness scoring ───────────────────────────────────────────────────────────

_FITNESS_CHECKS: dict[str, list[tuple[str, set[str]]]] = {
    "legal-compliance": [
        ("clause_type_identified", {
            "indemnif", "terminat", "liabilit", "warrant", "covenant",
            "confidential", "non-compete", "payment", "penalty",
            "arbitration", "force majeure", "intellectual property",
        }),
        ("risk_level_assigned", {
            "low risk", "medium risk", "high risk", "critical risk",
            "low-risk", "high-risk", "risk: low", "risk: medium",
            "risk: high", "risk: critical",
        }),
        ("obligation_extracted", {
            "shall ", "must ", "obligat", "required to", "responsible for",
            "duty to", "undertakes", "requires",
        }),
        ("party_identified", {
            "party", "parties", "licensee", "licensor", "buyer", "seller",
            "employer", "employee", "client", "vendor", "contractor",
        }),
    ],
    "finance-audit": [
        ("transaction_identified", {
            "transaction", "amount", "invoice", "payment", "receipt",
            "journal entry", "debit", "credit", "balance", "ledger",
        }),
        ("regulation_named", {
            "sox", "sarbanes", "aml", "anti-money laundering", "bsa",
            "bank secrecy", "fatf", "mifid", "sec ", "fca ", "pcaob",
        }),
        ("risk_flagged", {
            "anomaly", "suspicious", "unusual", "flag", "risk", "concern",
            "irregular", "discrepancy", "breach", "violation", "threshold",
        }),
        ("evidence_cited", {
            "ref:", "reference:", "transaction id", "txn", "doc:",
            "invoice #", "per ", "see ", "based on", "as per", "exhibit",
            "schedule", "line item", "control ref", "finding ref",
        }),
    ],
}


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
