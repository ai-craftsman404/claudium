"""Legal-compliance domain definition."""

from __future__ import annotations

from claudium.teams.domain import Domain

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

FITNESS_CHECKS: list[tuple[str, set[str]]] = [
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
]
