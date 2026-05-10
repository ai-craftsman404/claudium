"""Legal-compliance specialist definitions."""

from __future__ import annotations

from claudium.teams.specialist import Specialist

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

SPECIALISTS: list[Specialist] = [CLAUSE_EXTRACTOR, OBLIGATION_VALIDATOR, RISK_CLASSIFIER]
