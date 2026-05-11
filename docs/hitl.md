# Human-in-the-Loop (HITL)

Claudium v3d introduces a HITL checkpoint that pauses a team run after specialists complete and before LLM adjudication, allowing a human to approve or reject the run.

## How it works

1. Specialists run and produce findings
2. Rule-based fitness check runs (zero API cost)
3. **HITL checkpoint fires** — your callback receives an `ApprovalRequest`
4. If approved → LLM adjudication proceeds
5. If rejected → run stops, specialist results preserved

## Quick start

```python
import claudium_finance
from claudium.types import ApprovalRequest, ApprovalResponse
from claudium.teams.session import TeamSession

async def my_approval_handler(req: ApprovalRequest) -> ApprovalResponse:
    # Send Slack message, call webhook, display in UI, etc.
    print(f"Approval needed for run {req.run_id} in domain {req.domain}")
    for s in req.specialists:
        print(f"  {s.name}: score={s.fitness_score:.2f}")
    # Await human decision via your own mechanism
    approved = await ask_human(req.summary)
    return ApprovalResponse(approved=approved, reason="Reviewed by auditor")

ts = TeamSession(agent=agent, session_id="audit-1")
result = await ts.run_team_v3(
    "Review invoice INV-001 for Q1 compliance",
    domain="finance-audit",
    on_approval_required=my_approval_handler,
)

if result.approval_rejected:
    print("Run rejected — specialist results available for review")
    print(result.specialist_results)
else:
    print("Run completed:", result.synthesis)
```

## ApprovalRequest fields

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | Unique run identifier — use for idempotent webhook delivery |
| `session_id` | `str` | Session that fired the checkpoint |
| `domain` | `str` | Domain name (e.g. `"finance-audit"`) — use for routing to the right reviewer |
| `prompt` | `str` | Original prompt that initiated the run |
| `specialists` | `list[SpecialistSummary]` | One entry per specialist with `name`, `output`, `fitness_score` |
| `summary` | `str` | Plain-text join of all specialist outputs — use for simple notifications |
| `rule_check_passed` | `bool` | Whether rule-based fitness check passed — `False` means issues were found |
| `gaps` | `list[str]` | Missing criteria identified by rule-based check |
| `contradictions` | `list[str]` | Contradictions found by rule-based check |
| `created_at` | `str` | ISO 8601 UTC timestamp of when checkpoint fired |

## Checkpoint placement

The default (and only) checkpoint in v3d is `"post_specialists"`: fires after all specialists complete, before any LLM adjudication. This gives the human reviewer the rule-based findings at zero extra API cost.

```python
result = await ts.run_team_v3(
    prompt,
    domain="finance-audit",
    on_approval_required=handler,
    checkpoint="post_specialists",  # default — can be omitted
)
```

## Result fields

| Field | Description |
|---|---|
| `result.stop_reason` | `None` (normal), `"approval_rejected"`, or `"budget_exceeded"` |
| `result.approval_rejected` | `True` when human rejected |
| `result.truncated` | `True` when budget exceeded (deprecated — use `stop_reason`) |
| `result.specialist_results` | Always populated with pre-checkpoint results, even on rejection |

## Budget behaviour

The token budget is re-checked when the human approves and execution resumes. If the budget is exhausted during the approval window, `BudgetExceededError` is raised. Specialist results are preserved in the database — retrieve them by querying `team_runs_v3` with the `run_id` from `BudgetExceededError`.

## Testing with MockHarness

HITL is fully testable with zero API calls:

```python
async def auto_approve(req: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(approved=True)

ts = TeamSession(agent=agent, session_id="test-1")
result = await ts.run_team_v3(
    "Test prompt",
    domain="legal-compliance",
    on_approval_required=auto_approve,
)
assert result.stop_reason is None
```

## Error handling

If your callback raises an exception, Claudium preserves specialist results to the database before propagating the error:

```python
try:
    result = await ts.run_team_v3(..., on_approval_required=my_handler)
except Exception:
    # Specialist results are in the DB — query team_runs_v3 to retrieve them
    raise
```
