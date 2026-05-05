"""Orchestrator session: agent teams, parallel execution, consensus signals."""

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
from claudium.types import ConsensusSignal, HarnessResult, TeamResult

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


# ── SQLite helpers ────────────────────────────────────────────────────────────


async def _ensure_team_tables(db_path: Path) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists team_runs ("
            "id text primary key, "
            "prompt text not null, "
            "agent_count integer not null, "
            "agreement_score real not null, "
            "majority_output text, "
            "synthesis text, "
            "created_at text not null"
            ")"
        )
        await db.execute(
            "create table if not exists agent_outputs ("
            "id integer primary key autoincrement, "
            "run_id text not null references team_runs(id), "
            "agent_index integer not null, "
            "output_text text not null, "
            "is_outlier integer not null default 0"
            ")"
        )
        await db.commit()


async def _record_team_run(db_path: Path, result: TeamResult) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into team_runs values (?, ?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.prompt,
                len(result.outputs),
                result.consensus.agreement_score,
                result.consensus.majority_output,
                result.synthesis,
                datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            ),
        )
        for i, output in enumerate(result.outputs):
            await db.execute(
                "insert into agent_outputs(run_id, agent_index, output_text, is_outlier) "
                "values (?, ?, ?, ?)",
                (result.run_id, i, output.text, int(i in result.consensus.outlier_indices)),
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


class OrchestratorSession(ClaudiumSession):
    """ClaudiumSession with agent team management and consensus signal collection."""

    def __init__(
        self,
        *,
        agent: Any,
        session_id: str,
        role: str | None = None,
    ) -> None:
        super().__init__(agent=agent, session_id=session_id, role=role)
        self._team: list[ClaudiumTask] = []

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
    ) -> TeamResult:
        """Run the same prompt through all team members in parallel, return TeamResult."""
        if not self._team:
            raise RuntimeError("Call team() before run_team()")
        outputs: list[HarnessResult] = list(
            await asyncio.gather(*[task.prompt(prompt, model=model) for task in self._team])
        )
        run_id = str(uuid.uuid4())
        consensus = calculate_consensus(outputs)
        result = TeamResult(run_id=run_id, prompt=prompt, outputs=outputs, consensus=consensus)
        await _record_team_run(self.db_path, result)
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
