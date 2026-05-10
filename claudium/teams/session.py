"""TeamSession — v3 specialist team orchestration with domain-aware routing."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from claudium.orchestrator import OrchestratorSession
from claudium.teams.domain import DOMAINS, infer_domain, score_fitness
from claudium.teams.specialist import Specialist, select_specialists
from claudium.types import BudgetExceededError, HarnessResult

# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class SpecialistResult:
    specialist: Specialist
    output: HarnessResult
    fitness_score: float          # 0.0–1.0, domain-aware validation


@dataclass
class AdjudicationResult:
    mode: str                     # "rule-based" | "llm"
    accepted: bool
    gaps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    re_dispatch: list[str] = field(default_factory=list)  # specialist names
    synthesis: str | None = None


@dataclass
class TeamRunV3Result:
    run_id: str
    prompt: str
    domain: str
    specialist_results: list[SpecialistResult]
    adjudication: AdjudicationResult | None = None
    synthesis: str | None = None
    truncated: bool = False


# ── Complexity inference ──────────────────────────────────────────────────────


def _infer_complexity(prompt: str) -> int:
    """Heuristic complexity from prompt length: 1 simple / 2 moderate / 3 full team."""
    words = len(prompt.split())
    if words < 50:
        return 1
    if words < 150:
        return 2
    return 3


# ── Rule-based adjudication ───────────────────────────────────────────────────

_RISK_LEVELS = ["low risk", "medium risk", "high risk", "critical risk"]
_RISK_RANK = {level: i for i, level in enumerate(_RISK_LEVELS)}
_CONTRADICTION_SPAN = 2   # risk levels this far apart are contradictory


def _adjudicate_rule_based(
    specialist_results: list[SpecialistResult],
    domain: str,
    *,
    threshold: float = 0.75,
) -> AdjudicationResult:
    """Rule-based adjudication — fitness threshold, evidence check, contradiction detection."""
    gaps: list[str] = []
    contradictions: list[str] = []
    re_dispatch: list[str] = []

    for sr in specialist_results:
        if sr.fitness_score < threshold:
            gaps.append(
                f"{sr.specialist.name}: fitness {sr.fitness_score:.2f} below {threshold}"
            )
            re_dispatch.append(sr.specialist.name)

    # Contradiction detection — extreme risk level spread across specialists
    risk_found: list[tuple[str, str]] = []
    for sr in specialist_results:
        text = sr.output.text.lower()
        for level in _RISK_LEVELS:
            if level in text:
                risk_found.append((sr.specialist.name, level))
                break

    if len(risk_found) >= 2:
        ranks = [_RISK_RANK[level] for _, level in risk_found]
        if max(ranks) - min(ranks) >= _CONTRADICTION_SPAN:
            contradictions.append(
                "contradictory risk assessments: "
                + ", ".join(f"{name}={level}" for name, level in risk_found)
            )

    accepted = not gaps and not contradictions
    return AdjudicationResult(
        mode="rule-based",
        accepted=accepted,
        gaps=gaps,
        contradictions=contradictions,
        re_dispatch=re_dispatch,
    )


# ── SQLite helpers ────────────────────────────────────────────────────────────


async def _ensure_v3_tables(db_path: Any) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "create table if not exists domain_registry ("
            "id integer primary key autoincrement, "
            "session_id text not null, domain_name text not null, "
            "inferred_at text not null)"
        )
        await db.execute(
            "create table if not exists team_runs_v3 ("
            "id text primary key, prompt text not null, domain text not null, "
            "specialists_used text not null, adjudication text, "
            "synthesis text, created_at text not null)"
        )
        await db.execute(
            "create table if not exists specialist_runs ("
            "id integer primary key autoincrement, "
            "run_id text not null references team_runs_v3(id), "
            "specialist_name text not null, output_text text not null, "
            "fitness_score real not null)"
        )
        # Migrate existing databases — add adjudication column if absent
        try:
            await db.execute("alter table team_runs_v3 add column adjudication text")
        except Exception:
            pass
        await db.commit()


async def _persist_domain(db_path: Any, session_id: str, domain_name: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into domain_registry(session_id, domain_name, inferred_at) values (?,?,?)",
            (session_id, domain_name, datetime.now(timezone.utc).isoformat()),  # noqa: UP017
        )
        await db.commit()


async def _record_v3_run(db_path: Any, result: TeamRunV3Result) -> None:
    specialists_json = json.dumps(
        [sr.specialist.name for sr in result.specialist_results]
    )
    adj_json = None
    if result.adjudication:
        adj_json = json.dumps({
            "mode": result.adjudication.mode,
            "accepted": result.adjudication.accepted,
            "gaps": result.adjudication.gaps,
            "contradictions": result.adjudication.contradictions,
        })
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "insert into team_runs_v3"
            "(id, prompt, domain, specialists_used, adjudication, synthesis, created_at)"
            " values (?,?,?,?,?,?,?)",
            (
                result.run_id, result.prompt, result.domain,
                specialists_json, adj_json, result.synthesis,
                datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            ),
        )
        for sr in result.specialist_results:
            await db.execute(
                "insert into specialist_runs"
                "(run_id, specialist_name, output_text, fitness_score)"
                " values (?,?,?,?)",
                (result.run_id, sr.specialist.name, sr.output.text, sr.fitness_score),
            )
        await db.commit()


# ── TeamSession ───────────────────────────────────────────────────────────────


class TeamSession(OrchestratorSession):
    """OrchestratorSession extended with domain-aware specialist team routing."""

    async def _ensure_store(self) -> None:
        await super()._ensure_store()
        await _ensure_v3_tables(self.db_path)

    async def infer_domain(self, samples: list[str]) -> str:
        """Infer task domain from sample prompts. Persists result to SQLite."""
        domain_name = await infer_domain(
            samples,
            harness=self.agent.harness,
            config=self.agent.config,
            system_prompt=self.agent.instructions,
        )
        await _persist_domain(self.db_path, self.session_id, domain_name)
        return domain_name

    def score_fitness(self, text: str, domain: str) -> float:
        """Domain-aware fitness score for a specialist output."""
        return score_fitness(text, domain)

    async def run_specialists(
        self,
        prompt: str,
        specialists: list[Specialist],
        domain: str,
    ) -> tuple[list[SpecialistResult], bool]:
        """Run specialists; returns (results, truncated).

        truncated=True when the token budget was hit mid-run (sequential only).
        """
        domain_obj = DOMAINS.get(domain)
        if domain_obj and domain_obj.execution_strategy == "sequential":
            return await self._run_sequential(prompt, specialists, domain)
        return await self._run_parallel(prompt, specialists, domain), False

    async def _run_parallel(
        self,
        prompt: str,
        specialists: list[Specialist],
        domain: str,
    ) -> list[SpecialistResult]:
        async def _run_one(spec: Specialist) -> SpecialistResult:
            task = await self.task(spec.name)
            output = await task.prompt(f"{spec.instructions}\n\n{prompt}")
            return SpecialistResult(
                specialist=spec, output=output,
                fitness_score=score_fitness(output.text, domain),
            )
        results = await asyncio.gather(*[_run_one(s) for s in specialists])
        return list(results)

    async def _run_sequential(
        self,
        prompt: str,
        specialists: list[Specialist],
        domain: str,
    ) -> tuple[list[SpecialistResult], bool]:
        """Sequential execution — each specialist receives prior findings as context.

        Returns (results, truncated). truncated=True when budget exhausted mid-run.
        NOTE: specialist token usage is written to child task DBs, not the session DB,
        so _check_budget() here measures session-level tokens only (e.g. adjudication calls).
        """
        results: list[SpecialistResult] = []
        for spec in specialists:
            try:
                await self._check_budget()
            except BudgetExceededError:
                return results, True
            task = await self.task(spec.name)
            if results:
                prior = "\n\n".join(
                    f"{r.specialist.name} findings:\n{r.output.text}" for r in results
                )
                spec_prompt = (
                    f"{spec.instructions}\n\n"
                    f"Prior specialist findings:\n{prior}\n\n"
                    f"Original task:\n{prompt}"
                )
            else:
                spec_prompt = f"{spec.instructions}\n\n{prompt}"
            output = await task.prompt(spec_prompt)
            results.append(SpecialistResult(
                specialist=spec, output=output,
                fitness_score=score_fitness(output.text, domain),
            ))
        return results, False

    async def _adjudicate_llm(
        self,
        result: TeamRunV3Result,
        *,
        model: str | None = None,
    ) -> str:
        """LLM adjudication — domain-aware synthesis resolving gaps and contradictions."""
        outputs_text = "\n\n".join(
            f"{sr.specialist.name} (fitness: {sr.fitness_score:.2f}):\n{sr.output.text}"
            for sr in result.specialist_results
        )
        adj = result.adjudication
        issues = (adj.gaps + adj.contradictions) if adj else []
        issues_text = "\n".join(f"- {i}" for i in issues) if issues else "none"
        synthesis_prompt = (
            f"You are the orchestrator adjudicating specialist findings.\n\n"
            f"Domain: {result.domain}\n"
            f"Original task: {result.prompt}\n\n"
            f"Specialist outputs:\n{outputs_text}\n\n"
            f"Issues from rule-based adjudication:\n{issues_text}\n\n"
            f"Produce a coherent, evidence-based adjudicated finding that:\n"
            f"1. Resolves any contradictions between specialists\n"
            f"2. Fills identified gaps\n"
            f"3. Cites evidence for every conclusion\n"
            f"4. Follows {result.domain} domain standards"
        )
        config = self._effective_config(model, None)
        t0 = time.perf_counter()
        harness_result = await self.agent.harness.run(
            prompt=synthesis_prompt,
            system_prompt=self.agent.instructions,
            config=config,
        )
        await self._log_call(
            model=config.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=harness_result.raw,
            skill="adjudicate",
        )
        return harness_result.text

    async def run_team_v3(
        self,
        prompt: str,
        *,
        domain: str,
        complexity: int | None = None,
        model: str | None = None,
        adjudication_threshold: float = 0.75,
    ) -> TeamRunV3Result:
        """Run prompt through domain-appropriate specialist team with hybrid adjudication."""
        await self._ensure_store()
        if domain not in DOMAINS and domain != "unknown":
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {', '.join(DOMAINS)}"
            )
        inferred_complexity = complexity if complexity is not None else _infer_complexity(prompt)
        specialists = select_specialists(domain, complexity=inferred_complexity)

        truncated = False
        specialist_results: list[SpecialistResult] = []
        try:
            await self._check_budget()
            specialist_results, truncated = await self.run_specialists(
                prompt, specialists, domain
            )
        except BudgetExceededError:
            truncated = True

        result = TeamRunV3Result(
            run_id=str(uuid.uuid4()),
            prompt=prompt,
            domain=domain,
            specialist_results=specialist_results,
            truncated=truncated,
        )

        if truncated:
            await _record_v3_run(self.db_path, result)
            return result

        # Rule-based adjudication first (zero API cost)
        adj = _adjudicate_rule_based(
            specialist_results, domain, threshold=adjudication_threshold
        )
        result.adjudication = adj

        # Escalate to LLM adjudication only if rule-based didn't accept
        if not adj.accepted:
            synthesis = await self._adjudicate_llm(result, model=model)
            result.synthesis = synthesis
            adj.mode = "llm"
            adj.synthesis = synthesis

        await _record_v3_run(self.db_path, result)
        return result
