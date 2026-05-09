# Calibration

Claudium's self-improvement loop routes agent-team outputs through a weighted consensus
pipeline. The weights start neutral (1.0) and converge toward each agent's observed
agreement rate over time. Calibration kick-starts this process using your own sample data.

---

## What is a sample dataset?

A plain-text file with one representative prompt per line — the kind of inputs your skill
will see in production.

```
Triage issue: login fails on Safari macOS 14
Triage issue: 500 error from /api/checkout after DB migration
Triage issue: dark mode toggle doesn't persist across sessions
Triage issue: email verification link expires too fast
```

**Recommended size:** 20–50 samples minimum. Fewer gives noisy initial weights; more is
always better.

**Data quality is your responsibility.** Claudium provides the evaluation framework, not
the data. Representative, diverse samples produce better-calibrated weights than narrow or
synthetic ones.

---

## Running calibration

```bash
claudium calibrate triage --dataset samples/triage.txt
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` / `-d` | *(required)* | Path to dataset file |
| `--team-size` | `3` | Number of sub-agents in the team |
| `--window` | `10` | Rolling average window size |

Output:

```
Calibrating skill=triage samples=42 team=3
┏━━━━━━━┳━━━━━━━┳━━━━━━┓
┃ Agent ┃ Weight┃ Runs ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━┩
│ 0     │ 0.881 │ 10   │
│ 1     │ 0.952 │ 10   │
│ 2     │ 0.714 │ 10   │
└───────┴───────┴──────┘
Mean agreement: 0.85
```

---

## Cold start

Before any calibration, every agent weight is 1.0 (neutral). The routing pipeline still
works — it uses simple majority consensus — but weighted routing (level 3 of the
evaluation tree) has no signal yet.

Run calibration once before going to production for meaningful weight-based routing.

---

## When to re-calibrate

- After swapping models or changing roles
- After noticing a drop in consensus scores via `claudium trace`
- On a regular schedule (weekly or monthly, depending on traffic volume)
- After adding new agents to the team

Re-calibration is additive — it continues updating the rolling window rather than
resetting it from scratch.

---

## How weights update

After every `run_team()` call (including calibration), each agent's weight is updated via
a rolling mean:

```
new_weight = (old_weight × effective_count + agreed) / (effective_count + 1)
```

where `agreed = 1` if the agent matched the majority output, `0` otherwise, and
`effective_count = min(run_count, window - 1)`.

This means:
- Weights stay in [0.0, 1.0]
- Recent runs matter more than old ones (window caps the denominator)
- An agent that consistently agrees converges toward 1.0; one that consistently
  diverges converges toward 0.0

---

## Evaluation tree

Weights feed into level 3 of the 4-level routing pipeline:

```
Level 1 — Schema gate     (typed outputs only: fail-fast on Pydantic errors)
Level 2 — Consensus gate  (agreement ≥ 0.8 → return majority, no synthesis)
Level 3 — Weighted gate   (weight-adjusted confidence ≥ 0.6 → return weighted majority)
Level 4 — Synthesis       (real API call — only reached when lower levels don't resolve)
```

Well-calibrated weights push more calls through level 3, reducing synthesis API cost.
