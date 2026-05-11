"""Orchestrator session: agent teams, parallel execution, consensus signals, self-improvement."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from claudium.core import ClaudiumSession, ClaudiumTask
from claudium.harness.base import HarnessProtocol
from claudium.types import (
    AgentWeight,
    CalibrationResult,
    ConsensusSignal,
    HarnessResult,
    TeamResult,
)

# ── Consensus calculation ─────────────────────────────────────────────────────


def calculate_consensus(outputs: list[HarnessResult]) -> ConsensusSignal:
    """Compute agreement/disagreement across agent outputs — no ground truth needed."""
    if not outputs:
        return ConsensusSignal(agreement_score=0.0, majority_output=None, outlier_indices=[])

    normalised = [o.text.strip().lower() for o in outputs]
    counter: Counter[str] = Counter(normalised)
    mode_key, majority_count = counter.most_common(1)[0]
    agreement_score = majority_count / len(outputs)
    majority_output = next(o.text for o, n in zip(outputs, normalised) if n == mode_key)
    outlier_indices = [i for i, n in enumerate(normalised) if n != mode_key]

    return ConsensusSignal(
        agreement_score=agreement_score,
        majority_output=majority_output,
        outlier_indices=outlier_indices,
    )


def _weighted_confidence(
    outputs: list[HarnessResult],
    consensus: ConsensusSignal,
    weights: list[float],
) -> float:
    """Weighted confidence: max cluster weight sum / total weight."""
    total = sum(weights)
    if total == 0 or not outputs:
        return 0.0
    clusters: dict[str, float] = {}
    for i, output in enumerate(outputs):
        key = output.text.strip().lower()
        clusters[key] = clusters.get(key, 0.0) + weights[i]
    return max(clusters.values()) / total


# ── SQLite helpers ────────────────────────────────────────────────────────────


async def _ensure_team_tables(db_path: Path) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists team_runs ("
            "id text primary key, prompt text not null, agent_count integer not null, "
            "agreement_score real not null, majority_output text, synthesis text, "
            "created_at text not null, skill text, resolved_at text)"
        )
        await db.execute(
            "create table if not exists agent_outputs ("
            "id integer primary key autoincrement, "
            "run_id text not null references team_runs(id), "
            "agent_index integer not null, output_text text not null, "
            "is_outlier integer not null default 0)"
        )
        await db.execute(
            "create table if not exists agent_weights ("
            "id integer primary key autoincrement, skill text not null, "
            "agent_index integer not null, weight real not null default 1.0, "
            "run_count integer not null default 0, updated_at text not null, "
            "unique(skill, agent_index))"
        )
        await db.commit()


async def _record_team_run(db_path: Path, result: TeamResult) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into team_runs(id, prompt, agent_count, agreement_score, majority_output, "
            "synthesis, created_at, skill, resolved_at) values (?,?,?,?,?,?,?,?,?)",
            (
                result.run_id, result.prompt, len(result.outputs),
                result.consensus.agreement_score, result.consensus.majority_output,
                result.synthesis, datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                result.skill, result.resolved_at,
            ),
        )
        for i, output in enumerate(result.outputs):
            await db.execute(
                "insert into agent_outputs(run_id, agent_index, output_text, is_outlier) "
                "values (?, ?, ?, ?)",
                (result.run_id, i, output.text, int(i in result.consensus.outlier_indices)),
            )
        await db.commit()


async def _get_weights(db_path: Path, skill: str, n_agents: int) -> list[float]:
    """Return per-agent weights; defaults to 1.0 (neutral) for agents with no history."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "select agent_index, weight from agent_weights where skill=?", (skill,)
        )
        rows = await cursor.fetchall()
    weight_map = {int(r[0]): float(r[1]) for r in rows}
    return [weight_map.get(i, 1.0) for i in range(n_agents)]


async def _update_weights(
    db_path: Path,
    skill: str,
    consensus: ConsensusSignal,
    n_agents: int,
    window: int,
) -> None:
    """Rolling mean update: each agent's weight converges toward recent agreement rate."""
    async with aiosqlite.connect(db_path) as db:
        for i in range(n_agents):
            agreed = 0 if i in consensus.outlier_indices else 1
            cursor = await db.execute(
                "select weight, run_count from agent_weights where skill=? and agent_index=?",
                (skill, i),
            )
            row = await cursor.fetchone()
            if row is None:
                new_weight, new_count = float(agreed), 1
            else:
                old_weight, old_count = float(row[0]), int(row[1])
                effective = min(old_count, window - 1)
                new_weight = (old_weight * effective + agreed) / (effective + 1)
                new_count = min(old_count + 1, window)
            await db.execute(
                "insert or replace into agent_weights"
                "(skill, agent_index, weight, run_count, updated_at) values (?,?,?,?,?)",
                (skill, i, new_weight, new_count,
                 datetime.now(timezone.utc).isoformat()),  # noqa: UP017
            )
        await db.commit()


# ── Agent proxy ───────────────────────────────────────────────────────────────


class _AgentProxy:
    """Gives a sub-agent task its own harness without mutating the shared ClaudiumAgent."""

    def __init__(self, agent: Any, harness: HarnessProtocol) -> None:
        self._agent = agent
        self.harness = harness  # instance attr — shadows _agent.harness on lookup

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


# ── OrchestratorSession ───────────────────────────────────────────────────────

_HIGH_THRESHOLD = 0.8
_MID_THRESHOLD = 0.6


class OrchestratorSession(ClaudiumSession):
    """ClaudiumSession with agent team management, consensus signals, and self-improvement."""

    def __init__(
        self,
        *,
        agent: Any,
        session_id: str,
        role: str | None = None,
        weight_window: int = 10,
        high_threshold: float = _HIGH_THRESHOLD,
        mid_threshold: float = _MID_THRESHOLD,
        user_id: str | None = None,
        on_approval_required: Any = None,
    ) -> None:
        super().__init__(
            agent=agent, session_id=session_id, role=role,
            user_id=user_id, on_approval_required=on_approval_required,
        )
        self._team: list[ClaudiumTask] = []
        self._weight_window = weight_window
        self._high_threshold = high_threshold
        self._mid_threshold = mid_threshold

    async def _ensure_store(self) -> None:
        await super()._ensure_store()
        await _ensure_team_tables(self.db_path)

    async def team(
        self,
        n: int,
        *,
        role: str | None = None,
        harnesses: list[HarnessProtocol | None] | None = None,
    ) -> list[ClaudiumTask]:
        """Spawn N isolated sub-agent tasks, each optionally with its own harness."""
        tasks = []
        for i in range(n):
            task = await self.task(f"agent-{i}", role=role)
            if harnesses and i < len(harnesses) and harnesses[i] is not None:
                task.agent = _AgentProxy(self.agent, harnesses[i])  # type: ignore[assignment]
            tasks.append(task)
        self._team = tasks
        return tasks

    async def run_team(
        self,
        prompt: str,
        *,
        model: str | None = None,
        skill: str | None = None,
        auto_synthesise: bool = False,
    ) -> TeamResult:
        """Run prompt through team, apply evaluation tree, return TeamResult."""
        if not self._team:
            raise RuntimeError("Call team() before run_team()")
        outputs: list[HarnessResult] = list(
            await asyncio.gather(*[task.prompt(prompt, model=model) for task in self._team])
        )
        consensus = calculate_consensus(outputs)
        result = TeamResult(
            run_id=str(uuid.uuid4()), prompt=prompt, outputs=outputs,
            consensus=consensus, skill=skill,
        )

        # Level 2 — high consensus gate
        if consensus.agreement_score >= self._high_threshold:
            result.resolved_at = "consensus"
        elif skill:
            # Level 3 — weight-adjusted routing
            weights = await _get_weights(self.db_path, skill, len(self._team))
            if _weighted_confidence(outputs, consensus, weights) >= self._mid_threshold:
                result.resolved_at = "weighted"
            else:
                result.resolved_at = "synthesis_needed"
        else:
            result.resolved_at = "synthesis_needed"

        await _record_team_run(self.db_path, result)
        if skill:
            await _update_weights(
                self.db_path, skill, consensus, len(self._team), self._weight_window
            )

        # Level 4 — auto-synthesise if needed and requested
        if auto_synthesise and result.resolved_at == "synthesis_needed":
            await self.synthesise(result, model=model)

        return result

    async def synthesise(
        self,
        result: TeamResult,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> str:
        """Orchestrator synthesises team results — the one real API call in the pipeline."""
        outputs_text = "\n".join(
            f"Agent {i}: {o.text.strip()}" for i, o in enumerate(result.outputs)
        )
        synthesis_prompt = (
            f"You are the orchestrator. Synthesise {len(result.outputs)} agent responses.\n\n"
            f"Original prompt: {result.prompt}\n\n"
            f"Agent outputs:\n{outputs_text}\n\n"
            f"Agreement score: {result.consensus.agreement_score:.2f}\n"
            f"Majority: {result.consensus.majority_output or '(none)'}\n"
            f"Outliers: {result.consensus.outlier_indices or []}\n\n"
            "Provide the synthesised best answer."
        )
        config = self._effective_config(model, role)
        t0 = time.perf_counter()
        harness_result = await self.agent.harness.run(
            prompt=synthesis_prompt,
            system_prompt=self.agent.instructions,
            config=config,
        )
        await self._log_call(
            model=config.model, latency_ms=(time.perf_counter() - t0) * 1000,
            raw=harness_result.raw, skill="synthesise",
        )
        result.synthesis = harness_result.text
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "update team_runs set synthesis = ? where id = ?",
                (result.synthesis, result.run_id),
            )
            await db.commit()
        return harness_result.text

    async def calibrate(
        self,
        skill: str,
        samples: list[str],
        *,
        model: str | None = None,
    ) -> CalibrationResult:
        """Run orchestrator against sample dataset to initialise routing weights."""
        total_agreement = 0.0
        for sample in samples:
            result = await self.run_team(sample, model=model, skill=skill, auto_synthesise=False)
            total_agreement += result.consensus.agreement_score

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "select agent_index, weight, run_count, updated_at "
                "from agent_weights where skill=? order by agent_index",
                (skill,),
            )
            rows = await cursor.fetchall()

        agent_weights = [
            AgentWeight(
                skill=skill, agent_index=int(r[0]), weight=float(r[1]),
                run_count=int(r[2]), updated_at=str(r[3]),
            )
            for r in rows
        ]
        return CalibrationResult(
            skill=skill,
            samples_run=len(samples),
            weights=agent_weights,
            mean_agreement=total_agreement / len(samples) if samples else 0.0,
        )
