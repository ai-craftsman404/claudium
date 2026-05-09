"""Boundary and equivalence-class tests for domain scoring, complexity, and adjudication."""

from __future__ import annotations

from claudium.teams.domain import score_fitness
from claudium.teams.session import (
    SpecialistResult,
    _adjudicate_rule_based,
    _infer_complexity,
)
from claudium.teams.specialist import Specialist
from claudium.types import HarnessResult

# ── Fitness score boundaries (finance-audit domain) ────────────────────────────


def test_fitness_zero_criteria_met() -> None:
    """Text with no domain keywords yields fitness 0.0."""
    text = "This is completely unrelated to finance or auditing."
    result = score_fitness(text, "finance-audit")
    assert result == 0.0


def test_fitness_one_criterion_met() -> None:
    """Only transaction keyword identified yields 1/4 = 0.25."""
    text = "We reviewed the transaction dated 2025-01-15."
    result = score_fitness(text, "finance-audit")
    assert result == 0.25


def test_fitness_two_criteria_met() -> None:
    """Transaction + regulation keywords identified yields 2/4 = 0.50."""
    text = "The transaction violated SOX compliance requirements."
    result = score_fitness(text, "finance-audit")
    assert result == 0.50


def test_fitness_three_criteria_met() -> None:
    """Transaction + regulation + risk keywords identified yields 3/4 = 0.75."""
    text = "The transaction revealed an anomaly that violates SOX requirements."
    result = score_fitness(text, "finance-audit")
    assert result == 0.75


def test_fitness_all_criteria_met() -> None:
    """All four criteria met (transaction, regulation, risk, evidence) yields 1.0."""
    text = (
        "The transaction of $50,000 shows an anomaly that violates SOX regulations, "
        "per our audit findings documented in Exhibit A."
    )
    result = score_fitness(text, "finance-audit")
    assert result == 1.0


# ── Legal-compliance fitness boundaries ───────────────────────────────────────


def test_fitness_legal_compliance_zero() -> None:
    """Legal-compliance domain: no keywords yields 0.0."""
    text = "This document is unrelated to legal matters."
    result = score_fitness(text, "legal-compliance")
    assert result == 0.0


def test_fitness_legal_compliance_partial() -> None:
    """Legal-compliance: clause + risk level identified yields 0.50."""
    text = "The indemnification clause carries high risk."
    result = score_fitness(text, "legal-compliance")
    assert result == 0.50


def test_fitness_legal_compliance_full() -> None:
    """Legal-compliance: clause, risk, obligation, party all present yields 1.0."""
    text = (
        "The indemnification clause requires the licensor to assume liability. "
        "This obligation carries medium risk for both parties."
    )
    result = score_fitness(text, "legal-compliance")
    assert result == 1.0


# ── Complexity inference boundaries ──────────────────────────────────────────


def test_complexity_at_49_words() -> None:
    """Prompt with 49 words (just below threshold) yields complexity 1."""
    prompt = " ".join(["word"] * 49)
    assert _infer_complexity(prompt) == 1


def test_complexity_at_50_words() -> None:
    """Prompt with 50 words (at boundary) yields complexity 2."""
    prompt = " ".join(["word"] * 50)
    assert _infer_complexity(prompt) == 2


def test_complexity_at_51_words() -> None:
    """Prompt with 51 words (just above first boundary) yields complexity 2."""
    prompt = " ".join(["word"] * 51)
    assert _infer_complexity(prompt) == 2


def test_complexity_at_149_words() -> None:
    """Prompt with 149 words (just below second boundary) yields complexity 2."""
    prompt = " ".join(["word"] * 149)
    assert _infer_complexity(prompt) == 2


def test_complexity_at_150_words() -> None:
    """Prompt with 150 words (at second boundary) yields complexity 3."""
    prompt = " ".join(["word"] * 150)
    assert _infer_complexity(prompt) == 3


def test_complexity_at_151_words() -> None:
    """Prompt with 151 words (above second boundary) yields complexity 3."""
    prompt = " ".join(["word"] * 151)
    assert _infer_complexity(prompt) == 3


def test_complexity_long_prompt() -> None:
    """Very long prompt (300+ words) yields complexity 3."""
    prompt = " ".join(["word"] * 300)
    assert _infer_complexity(prompt) == 3


# ── Helper for specialist results ────────────────────────────────────────────


def _sr(name: str, text: str, fitness: float) -> SpecialistResult:
    """Create a SpecialistResult for testing."""
    spec = Specialist(
        name=name,
        domain="finance-audit",
        focus="",
        instructions="",
    )
    return SpecialistResult(
        specialist=spec,
        output=HarnessResult(text=text),
        fitness_score=fitness,
    )


# ── Adjudication threshold boundaries ────────────────────────────────────────


def test_adjudication_accepts_at_threshold_075() -> None:
    """SpecialistResult with fitness=0.75 (at threshold) is accepted."""
    results = [_sr("auditor-1", "Transaction reviewed.", 0.75)]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is True
    assert adj.gaps == []
    assert adj.contradictions == []


def test_adjudication_rejects_below_threshold_074() -> None:
    """SpecialistResult with fitness=0.74 (below threshold) is rejected."""
    results = [_sr("auditor-1", "Transaction reviewed.", 0.74)]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is False
    assert len(adj.gaps) == 1
    assert "auditor-1" in adj.gaps[0]
    assert "0.74" in adj.gaps[0]


def test_adjudication_accepts_above_threshold_076() -> None:
    """SpecialistResult with fitness=0.76 (above threshold) is accepted."""
    results = [_sr("auditor-1", "Transaction reviewed.", 0.76)]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is True
    assert adj.gaps == []


def test_adjudication_custom_threshold() -> None:
    """Adjudication respects custom threshold parameter."""
    results = [_sr("auditor-1", "Transaction reviewed.", 0.50)]
    # With higher threshold, rejected
    adj_high = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj_high.accepted is False
    # With lower threshold, accepted
    adj_low = _adjudicate_rule_based(results, "finance-audit", threshold=0.40)
    assert adj_low.accepted is True


def test_adjudication_multiple_specialists_all_pass() -> None:
    """Multiple specialists all above threshold yields acceptance."""
    results = [
        _sr("auditor-1", "Transaction reviewed.", 0.80),
        _sr("risk-analyst", "Risk flagged.", 0.75),
        _sr("compliance-checker", "SOX compliant.", 0.90),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is True
    assert adj.gaps == []


def test_adjudication_multiple_specialists_one_fails() -> None:
    """Multiple specialists but one below threshold yields rejection."""
    results = [
        _sr("auditor-1", "Transaction reviewed.", 0.80),
        _sr("risk-analyst", "Risk flagged.", 0.60),  # Below threshold
        _sr("compliance-checker", "SOX compliant.", 0.90),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is False
    assert len(adj.gaps) == 1
    assert "risk-analyst" in adj.gaps[0]
    assert "risk-analyst" in adj.re_dispatch


# ── Contradiction detection: risk level span boundaries ──────────────────────


def test_no_contradiction_span_0() -> None:
    """Risk levels identical (span=0) produces no contradiction."""
    results = [
        _sr("auditor-1", "Risk assessment: low risk", 0.80),
        _sr("auditor-2", "Risk assessment: low risk", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.contradictions == []


def test_no_contradiction_span_1() -> None:
    """Risk levels adjacent (span=1: 'low risk' vs 'medium risk') produces no contradiction."""
    results = [
        _sr("auditor-1", "This is low risk.", 0.80),
        _sr("auditor-2", "This is medium risk.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.contradictions == []


def test_contradiction_at_span_2() -> None:
    """Risk levels span=2 ('low risk' vs 'high risk') triggers contradiction."""
    results = [
        _sr("auditor-1", "This is low risk.", 0.80),
        _sr("auditor-2", "This is high risk.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert len(adj.contradictions) == 1
    assert "contradictory risk assessments" in adj.contradictions[0]
    assert "low risk" in adj.contradictions[0]
    assert "high risk" in adj.contradictions[0]


def test_contradiction_at_span_3() -> None:
    """Risk levels span=3 ('low risk' vs 'critical risk') triggers contradiction."""
    results = [
        _sr("auditor-1", "This is low risk.", 0.80),
        _sr("auditor-2", "This is critical risk.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert len(adj.contradictions) == 1
    assert "contradictory risk assessments" in adj.contradictions[0]


def test_contradiction_medium_to_critical() -> None:
    """Risk levels 'medium risk' vs 'critical risk' (span=2) triggers contradiction."""
    results = [
        _sr("auditor-1", "Assessment: medium risk", 0.80),
        _sr("auditor-2", "Assessment: critical risk", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert len(adj.contradictions) == 1


def test_no_contradiction_single_specialist() -> None:
    """Single specialist (no pair) produces no contradiction detection."""
    results = [_sr("auditor-1", "Assessment: high risk", 0.80)]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.contradictions == []


def test_contradiction_with_three_specialists() -> None:
    """Three specialists where two span >= 2 levels triggers contradiction."""
    results = [
        _sr("auditor-1", "Assessment: low risk", 0.80),
        _sr("auditor-2", "Assessment: medium risk", 0.80),
        _sr("auditor-3", "Assessment: high risk", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    # Max rank=2 (high risk), min rank=0 (low risk), span=2 -> contradiction
    assert len(adj.contradictions) == 1


def test_contradiction_no_risk_level_mentioned() -> None:
    """When no risk level is mentioned, no contradiction is detected."""
    results = [
        _sr("auditor-1", "Transaction reviewed without risk assessment.", 0.80),
        _sr("auditor-2", "Compliance status verified without risk assessment.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.contradictions == []


def test_contradiction_partial_risk_levels() -> None:
    """Only one specialist mentions risk level; no contradiction."""
    results = [
        _sr("auditor-1", "Assessment: high risk", 0.80),
        _sr("auditor-2", "Findings without risk classification.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.contradictions == []


# ── Combined scenarios: threshold + contradiction ────────────────────────────


def test_adjudication_fails_on_gap_not_contradiction() -> None:
    """Fails due to low fitness (gap), not contradiction."""
    results = [
        _sr("auditor-1", "Low fitness output.", 0.50),
        _sr("auditor-2", "This is low risk.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is False
    assert len(adj.gaps) == 1
    assert adj.contradictions == []


def test_adjudication_fails_on_contradiction_not_gap() -> None:
    """Fails due to contradictory risk levels, not gaps."""
    results = [
        _sr("auditor-1", "This is low risk.", 0.90),
        _sr("auditor-2", "This is critical risk.", 0.90),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is False
    assert len(adj.contradictions) == 1
    assert adj.gaps == []


def test_adjudication_fails_on_both_gap_and_contradiction() -> None:
    """Fails due to both gap and contradiction."""
    results = [
        _sr("auditor-1", "Low fitness.", 0.50),
        _sr("auditor-2", "This is low risk.", 0.90),
        _sr("auditor-3", "This is critical risk.", 0.90),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.accepted is False
    assert len(adj.gaps) >= 1
    assert len(adj.contradictions) >= 1


# ── Edge cases and special scenarios ──────────────────────────────────────────


def test_fitness_case_insensitive() -> None:
    """Fitness scoring is case-insensitive."""
    text_lower = "transaction identified with sox violation and anomaly per audit"
    text_upper = "TRANSACTION IDENTIFIED WITH SOX VIOLATION AND ANOMALY PER AUDIT"
    text_mixed = "Transaction IDENTIFIED with SoX violation AND anomaly PER audit"
    assert score_fitness(text_lower, "finance-audit") == score_fitness(text_upper, "finance-audit")
    assert score_fitness(text_lower, "finance-audit") == score_fitness(text_mixed, "finance-audit")


def test_fitness_partial_keyword_matches() -> None:
    """Fitness detects partial keyword matches (e.g., 'sarbanes' contains 'sarbanes')."""
    text = "Sarbanes-Oxley compliance breach detected."
    # "sarbanes" is in the regulation keywords
    result = score_fitness(text, "finance-audit")
    assert result >= 0.25  # Should match at least regulation criterion


def test_complexity_empty_prompt() -> None:
    """Empty prompt (0 words) yields complexity 1."""
    assert _infer_complexity("") == 1


def test_complexity_whitespace_only() -> None:
    """Whitespace-only prompt (0 words) yields complexity 1."""
    assert _infer_complexity("   \n\t  ") == 1


def test_adjudication_empty_results() -> None:
    """Empty specialist results list: accepted if no gaps/contradictions."""
    adj = _adjudicate_rule_based([], "finance-audit", threshold=0.75)
    assert adj.accepted is True
    assert adj.gaps == []
    assert adj.contradictions == []


def test_adjudication_re_dispatch_populated() -> None:
    """Re-dispatch list populated when specialist below threshold."""
    results = [
        _sr("auditor-1", "Output.", 0.50),
        _sr("auditor-2", "Output.", 0.80),
    ]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert "auditor-1" in adj.re_dispatch
    assert "auditor-2" not in adj.re_dispatch


def test_adjudication_mode_is_rule_based() -> None:
    """Adjudication mode is set to 'rule-based' by default."""
    results = [_sr("auditor-1", "Output.", 0.80)]
    adj = _adjudicate_rule_based(results, "finance-audit", threshold=0.75)
    assert adj.mode == "rule-based"
