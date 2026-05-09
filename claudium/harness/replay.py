"""ReplayHarness — records production (prompt, response) pairs for deterministic replay."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from claudium.types import ClaudiumEvent, HarnessResult


class ReplayHarness:
    """
    Production: records every (prompt, response) pair to SQLite.
    Replay: replays any historical run deterministically from stored fixtures.

    Usage — production recording:
        harness = ReplayHarness(db_path, record=True)

    Usage — deterministic replay (regulatory audit, CI):
        harness = ReplayHarness(db_path, record=False)

    Usage — testing (MockHarness as backing):
        harness = ReplayHarness(db_path, record=True, backing_harness=mock)
    """

    def __init__(
        self,
        db_path: Path,
        *,
        record: bool = True,
        backing_harness: Any = None,
    ) -> None:
        self.db_path = db_path
        self.record = record
        self._backing = backing_harness

    async def _ensure_table(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "create table if not exists replay_log ("
                "id integer primary key autoincrement, "
                "prompt_hash text not null, prompt text not null, "
                "response text not null, recorded_at text not null)"
            )
            await db.commit()

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str,
        config: Any,
        result_tool: Any = None,
        tools: Any = None,
    ) -> HarnessResult:
        await self._ensure_table()
        prompt_hash = self._hash(prompt)

        if not self.record:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "select response from replay_log"
                    " where prompt_hash=? order by id desc limit 1",
                    (prompt_hash,),
                )
                row = await cursor.fetchone()
            if row is None:
                raise KeyError(
                    f"No recorded response for prompt hash '{prompt_hash}'. "
                    "Run in record mode first to capture fixtures."
                )
            return HarnessResult(text=str(row[0]))

        # Record mode — delegate to backing harness, store result
        backing = self._backing
        if backing is None:
            from claudium.harness.anthropic import AnthropicHarness
            backing = AnthropicHarness()

        result = await backing.run(
            prompt=prompt,
            system_prompt=system_prompt,
            config=config,
            result_tool=result_tool,
            tools=tools,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "insert into replay_log(prompt_hash, prompt, response, recorded_at)"
                " values (?,?,?,?)",
                (
                    prompt_hash, prompt, result.text,
                    datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                ),
            )
            await db.commit()
        return result

    async def stream(self, **_: Any) -> AsyncIterator[ClaudiumEvent]:
        yield ClaudiumEvent(type="text_delta", data={"text": ""})
