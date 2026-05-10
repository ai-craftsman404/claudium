"""Finance-audit domain definition."""

from __future__ import annotations

from claudium.teams.domain import Domain

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

FITNESS_CHECKS: list[tuple[str, set[str]]] = [
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
]
