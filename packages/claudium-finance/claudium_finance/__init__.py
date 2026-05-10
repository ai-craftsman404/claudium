"""Claudium Finance — finance-audit domain pack.

Importing this package registers the finance-audit domain and specialists
with Claudium core automatically::

    import claudium_finance
    from claudium.teams.session import TeamSession

    ts = TeamSession(agent=agent, session_id="audit-1")
    result = await ts.run_team_v3("Review invoice INV-001", domain="finance-audit")
"""

from claudium.teams.domain import register_domain
from claudium.teams.specialist import register_specialists
from claudium_finance.domain import FINANCE_AUDIT, FITNESS_CHECKS
from claudium_finance.specialist import SPECIALISTS

register_domain(FINANCE_AUDIT, FITNESS_CHECKS)
register_specialists("finance-audit", SPECIALISTS)

__all__ = ["FINANCE_AUDIT", "FITNESS_CHECKS", "SPECIALISTS"]
