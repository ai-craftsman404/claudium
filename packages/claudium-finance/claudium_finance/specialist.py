"""Finance-audit specialist definitions."""

from __future__ import annotations

from claudium.teams.specialist import Specialist

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

SPECIALISTS: list[Specialist] = [TRANSACTION_AUDITOR, RISK_ANALYST, COMPLIANCE_CHECKER]
