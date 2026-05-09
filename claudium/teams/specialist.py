"""Specialist agent definitions for v3 domains."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Specialist:
    name: str
    domain: str
    focus: str
    instructions: str


# ── Legal/Compliance specialist pool ──────────────────────────────────────────

CLAUSE_EXTRACTOR = Specialist(
    name="clause-extractor",
    domain="legal-compliance",
    focus="Identify and extract specific clause types from legal documents",
    instructions=(
        "You are a legal clause extraction specialist. "
        "For each clause you identify, provide: clause type, verbatim or summarised text, "
        "and the parties it binds. Be precise and exhaustive. "
        "Always name the clause type explicitly."
    ),
)

OBLIGATION_VALIDATOR = Specialist(
    name="obligation-validator",
    domain="legal-compliance",
    focus="Validate that all obligations have clear parties, deadlines, and conditions",
    instructions=(
        "You are a legal obligation validation specialist. "
        "For each obligation found, identify: the obligated party, the specific duty "
        "(using 'shall' or 'must' language), any deadline or trigger condition, "
        "and whether the obligation is clearly enforceable."
    ),
)

RISK_CLASSIFIER = Specialist(
    name="risk-classifier",
    domain="legal-compliance",
    focus="Classify the risk level of contract clauses and obligations",
    instructions=(
        "You are a legal risk classification specialist. "
        "Assess each clause or obligation and assign a risk level: "
        "low risk, medium risk, high risk, or critical risk. "
        "Provide a one-sentence justification for each classification."
    ),
)

# ── Finance/Audit specialist pool ─────────────────────────────────────────────

TRANSACTION_AUDITOR = Specialist(
    name="transaction-auditor",
    domain="finance-audit",
    focus="Review transactions for accuracy, completeness, and proper authorisation",
    instructions=(
        "You are a financial transaction auditor. "
        "For each transaction or journal entry, verify: amount, date, party, "
        "authorisation trail, and three-way matching (PO/invoice/receipt where applicable). "
        "Reference the specific transaction ID or document in every finding. "
        "Flag any missing fields or authorisation gaps explicitly."
    ),
)

RISK_ANALYST = Specialist(
    name="risk-analyst",
    domain="finance-audit",
    focus="Identify and classify financial risk, anomalies, and threshold breaches",
    instructions=(
        "You are a financial risk analyst. "
        "Based on the transaction findings provided, identify: anomalies, "
        "unusual patterns, concentration risk, and threshold breaches. "
        "Assign risk level (low risk, medium risk, high risk, critical risk) "
        "with quantitative justification. Cite the specific transaction reference "
        "or prior finding that supports each risk assessment."
    ),
)

COMPLIANCE_CHECKER = Specialist(
    name="compliance-checker",
    domain="finance-audit",
    focus="Verify adherence to SOX, AML, BSA regulations and internal control requirements",
    instructions=(
        "You are a financial compliance specialist. "
        "Based on the transaction audit and risk findings provided, verify compliance with: "
        "SOX internal control requirements, AML suspicious activity indicators (FATF typologies), "
        "and BSA reporting thresholds. "
        "For each compliance issue, cite: the specific regulation or control reference, "
        "the transaction or finding it relates to, and the required remediation action."
    ),
)

# ── Domain specialist pools ───────────────────────────────────────────────────

_DOMAIN_POOLS: dict[str, list[Specialist]] = {
    "legal-compliance": [CLAUSE_EXTRACTOR, OBLIGATION_VALIDATOR, RISK_CLASSIFIER],
    "finance-audit": [TRANSACTION_AUDITOR, RISK_ANALYST, COMPLIANCE_CHECKER],
}


def pool_for(domain: str) -> list[Specialist]:
    """Return the full specialist pool for a domain."""
    return list(_DOMAIN_POOLS.get(domain, []))


def select_specialists(domain: str, *, complexity: int = 1) -> list[Specialist]:
    """Select N specialists from the domain pool based on task complexity (1–3)."""
    pool = pool_for(domain)
    n = max(1, min(complexity, len(pool)))
    return pool[:n]
