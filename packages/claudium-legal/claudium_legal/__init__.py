"""Claudium Legal — legal-compliance domain pack.

Importing this package registers the legal-compliance domain and specialists
with Claudium core automatically::

    import claudium_legal
    from claudium.teams.session import TeamSession

    ts = TeamSession(agent=agent, session_id="review-1")
    result = await ts.run_team_v3("Review NDA clause 4.2", domain="legal-compliance")
"""

from claudium.teams.domain import register_domain
from claudium.teams.specialist import register_specialists
from claudium_legal.domain import FITNESS_CHECKS, LEGAL_COMPLIANCE
from claudium_legal.specialist import SPECIALISTS

register_domain(LEGAL_COMPLIANCE, FITNESS_CHECKS)
register_specialists("legal-compliance", SPECIALISTS)

__all__ = ["FITNESS_CHECKS", "LEGAL_COMPLIANCE", "SPECIALISTS"]
