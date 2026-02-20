"""EcologicalAgents 2-phase concurrent tick engine.

PHASE A — PLAN
  All agent decision prompts are built simultaneously.
  For LLM-eligible agents, async HTTP requests are dispatched concurrently
  via asyncio + aiohttp. Heuristic agents resolve immediately (no I/O).
  A per-tick deadline ensures no agent blocks the tick.
  Timed-out / failed agents fall back to the heuristic policy.

PHASE B — ACT
  Decisions are applied in a deterministic order (sorted by agent_id).
  All world state mutations happen here, after all decisions are collected.
  Conflict resolution, entropy updates, compute tracking, tool gating,
  irreversible mutations, and opportunity deadline checks all live in ACT.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import aiohttp

from lab_sim.agents.cognition import AgentAction, AgentCognition
from lab_sim.config.settings import TickEngineSettings
from lab_sim.utils.types import AgentState

if TYPE_CHECKING:
    from lab_sim.world.simulator import WorldSimulator

logger = logging.getLogger("lab_sim.engine")


# ---------------------------------------------------------------------------
# Decision result — carries provenance for logging
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    agent_name: str
    action: AgentAction
    latency_ms: float = 0.0
    used_heuristic: bool = False
    fallback_reason: str = ""
    compute_consumed: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0


# ---------------------------------------------------------------------------
# TickEngine
# ---------------------------------------------------------------------------

class TickEngine:
    """Orchestrates the 2-phase tick: PLAN (concurrent) then ACT (deterministic)."""

    def __init__(
        self,
        cfg: TickEngineSettings,
        cognitions: dict[str, AgentCognition],
        mode_set: set[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.cognitions = cognitions
        self.mode_set: set[str] = mode_set or set()
        # Persistent event loop — avoids creating/destroying one every tick
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Semaphore bound to this loop; created once
        self._llm_semaphore = asyncio.Semaphore(cfg.max_concurrent_llm)

    # ===================================================================
    # Public interface — called from WorldSimulator._run_tick()
    # ===================================================================

    def run_phases(
        self,
        tick: int,
        run_id: str,
        states: dict[str, AgentState],
        inboxes: dict[str, list[str]],
        peers_fn: Callable[[str], list[str]],
    ) -> list[DecisionResult]:
        """Entry point: runs PLAN phase on an event loop, returns sorted decisions."""
        alive = [name for name, s in states.items() if s.alive]
        if not alive:
            return []

        t0 = time.perf_counter()
        logger.info("PLAN start tick=%d agents=%d", tick, len(alive))

        # PLAN phase — run on the persistent loop (avoids loop creation overhead)
        results = self._loop.run_until_complete(
            self._plan_phase(tick, run_id, alive, states, inboxes, peers_fn)
        )

        elapsed = time.perf_counter() - t0
        llm_count = sum(1 for r in results if not r.used_heuristic)
        heuristic_count = len(results) - llm_count
        logger.info(
            "PLAN done  tick=%d elapsed=%.1fs llm=%d heuristic=%d",
            tick, elapsed, llm_count, heuristic_count,
        )

        # Sort deterministically — actions always applied in agent_id order
        results.sort(key=lambda r: r.agent_name)
        return results

    # ===================================================================
    # PHASE A — PLAN (concurrent decision collection)
    # ===================================================================

    async def _plan_phase(
        self,
        tick: int,
        run_id: str,
        alive: list[str],
        states: dict[str, AgentState],
        inboxes: dict[str, list[str]],
        peers_fn: Callable[[str], list[str]],
    ) -> list[DecisionResult]:
        """Dispatch all agent decisions concurrently with a per-tick deadline."""
        deadline = self.cfg.per_tick_deadline_s
        tasks: list[asyncio.Task] = []

        # force_close=True: each request gets a fresh connection — prevents
        # ServerDisconnectedError from Ollama closing idle/queued connections.
        connector = aiohttp.TCPConnector(limit=self.cfg.max_concurrent_llm, force_close=True)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"Connection": "close"},
        ) as session:
            for name in alive:
                state = states[name]
                peers = peers_fn(name)
                if not peers:
                    continue
                inbox = inboxes.get(name, [])
                cog = self.cognitions[name]
                task = asyncio.create_task(
                    self._collect_decision(
                        run_id=run_id,
                        tick=tick,
                        name=name,
                        state=state,
                        inbox=inbox,
                        peers=peers,
                        cog=cog,
                        session=session,
                    ),
                    name=f"decide_{name}_t{tick}",
                )
                tasks.append(task)

            if not tasks:
                return []

            # Wait for all tasks up to the per-tick deadline
            try:
                done, pending = await asyncio.wait(
                    tasks, timeout=deadline, return_when=asyncio.ALL_COMPLETED
                )
            except Exception as exc:
                logger.error("PLAN phase wait error tick=%d: %s", tick, exc)
                done, pending = set(), set(tasks)

        # Cancel any tasks that ran over deadline
        results: list[DecisionResult] = []
        for task in pending:
            task_name = task.get_name()
            # Task name format: "decide_{agent_name}_t{tick}"
            # Agent names may themselves contain underscores (e.g. "agent_00"),
            # so strip the known prefix and suffix rather than splitting naively.
            agent_name = "unknown"
            if task_name.startswith("decide_"):
                without_prefix = task_name[len("decide_"):]
                # Suffix is "_t{tick}" — find the last "_t" occurrence
                last_t = without_prefix.rfind("_t")
                if last_t != -1:
                    agent_name = without_prefix[:last_t]
                else:
                    agent_name = without_prefix
            logger.warning(
                "PLAN deadline reached: agent=%s tick=%d — heuristic fallback",
                agent_name, tick,
            )
            task.cancel()
            state = states.get(agent_name)
            cog = self.cognitions.get(agent_name)
            if state and cog:
                inbox = inboxes.get(agent_name, [])
                peers = peers_fn(agent_name)
                fallback_action = cog._heuristic_policy(inbox, peers)
                results.append(DecisionResult(
                    agent_name=agent_name,
                    action=fallback_action,
                    used_heuristic=True,
                    fallback_reason="deadline",
                    compute_consumed=0.0,
                    entropy_before=state.memory_entropy,
                    entropy_after=state.memory_entropy,
                ))

        for task in done:
            try:
                result = task.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.error("PLAN task exception tick=%d: %s", tick, exc)

        return results

    async def _collect_decision(
        self,
        run_id: str,
        tick: int,
        name: str,
        state: AgentState,
        inbox: list[str],
        peers: list[str],
        cog: AgentCognition,
        session: "aiohttp.ClientSession",
    ) -> DecisionResult:
        """Collect one agent's decision. Uses async LLM if budget allows."""
        entropy_before = state.memory_entropy
        compute_consumed = 0.0

        # Gate: if compute_budget exhausted, use heuristic immediately
        # (unless no_compute_pressure is active — infinite budget mode)
        if "no_compute_pressure" not in self.mode_set:
            if state.compute_exhausted or state.compute_budget <= self.cfg.compute_min_threshold:
                state.compute_exhausted = True
                action = cog._heuristic_policy(inbox, peers)
                logger.debug(
                    "compute_exhausted agent=%s tick=%d budget=%.3f — heuristic fallback",
                    name, tick, state.compute_budget,
                )
                return DecisionResult(
                    agent_name=name,
                    action=action,
                    used_heuristic=True,
                    fallback_reason="compute_exhausted",
                    compute_consumed=0.0,
                    entropy_before=entropy_before,
                    entropy_after=state.memory_entropy,
                )

        # Gate: if tool access for data_retrieval is revoked, skip memory retrieval
        can_retrieve = state.tool_access_flags.get("data_retrieval", True)

        # Decide whether LLM is warranted
        should_llm = cog._llm_should_trigger(tick, inbox)

        if not should_llm:
            action = cog._heuristic_policy(inbox, peers)
            logger.info("PLAN agent=%-12s tick=%d  HEURISTIC  reason=trigger_not_met", name, tick)
            return DecisionResult(
                agent_name=name,
                action=action,
                used_heuristic=True,
                fallback_reason="trigger_not_met",
                compute_consumed=0.0,
                entropy_before=entropy_before,
                entropy_after=state.memory_entropy,
            )

        # Memory retrieval (costs compute)
        recalled = []
        if can_retrieve:
            try:
                t_mem = time.perf_counter()
                memory_query = f"{state.goals} {inbox} {state.identity_signature()}"
                recalled = cog.memory.retrieve(run_id, name, memory_query, tick, limit=8)
                mem_ms = (time.perf_counter() - t_mem) * 1000.0
                compute_consumed += self.cfg.compute_per_retrieval
                state.compute_budget = max(0.0, state.compute_budget - self.cfg.compute_per_retrieval)
                if mem_ms > 2000:
                    logger.info("PLAN agent=%-12s tick=%d  MEM-RETRIEVE  %.0fms  hits=%d  (embed model cold-start)",
                                name, tick, mem_ms, len(recalled))
                else:
                    logger.info("PLAN agent=%-12s tick=%d  MEM-RETRIEVE  %.0fms  hits=%d",
                                name, tick, mem_ms, len(recalled))
            except Exception as exc:
                logger.warning("Memory retrieval failed agent=%s: %s", name, exc)

        # Build prompt
        prompt = cog._build_prompt(inbox, peers, recalled)
        prompt_tokens_approx = len(prompt) // 4  # rough estimate

        # Async LLM call — logs LLM-QUEUE now; OLLAMA request log fires after semaphore acquired
        logger.info("PLAN agent=%-12s tick=%d  LLM-QUEUE   prompt~%dtok  budget=%.2f",
                    name, tick, prompt_tokens_approx, state.compute_budget)
        logger.debug("PLAN agent=%s tick=%d  PROMPT:\n%s", name, tick, prompt)
        t0 = time.perf_counter()
        try:
            raw, latency_ms = await cog.ollama.async_generate(
                prompt,
                timeout_s=self.cfg.per_request_timeout_s,
                semaphore=self._llm_semaphore,
                session=session,
            )
            state.decision_latency_ms = latency_ms
            compute_consumed += self.cfg.compute_per_llm_call
            state.compute_budget = max(0.0, state.compute_budget - self.cfg.compute_per_llm_call)

            logger.info(
                "PLAN agent=%-12s tick=%d  LLM-DONE    latency=%.0fms  budget=%.2f  action=%s",
                name, tick, latency_ms, state.compute_budget,
                cog._parse_action(raw, peers).action if raw else "?",
            )

            # Apply entropy-based hallucination distortion (skip if no_entropy)
            if "no_entropy" not in self.mode_set:
                hallucination_prob = 0.05 + self.cfg.entropy_hallucination_scale * state.memory_entropy
                if cog.rng.random() < hallucination_prob:
                    raw = self._apply_entropy_distortion(raw, cog, name, tick)

            action = cog._parse_action(raw, peers)
            used_heuristic = False
            fallback_reason = ""

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            state.decision_latency_ms = latency_ms
            logger.warning(
                "PLAN agent=%-12s tick=%d  LLM-FAIL    latency=%.0fms  error=%s — heuristic fallback",
                name, tick, latency_ms, exc.__class__.__name__,
            )
            action = cog._heuristic_policy(inbox, peers)
            used_heuristic = True
            fallback_reason = f"llm_error:{exc.__class__.__name__}"

        # Mark exhausted if budget fell below threshold
        if state.compute_budget <= self.cfg.compute_min_threshold:
            state.compute_exhausted = True

        return DecisionResult(
            agent_name=name,
            action=action,
            latency_ms=latency_ms if not used_heuristic else 0.0,
            used_heuristic=used_heuristic,
            fallback_reason=fallback_reason,
            compute_consumed=compute_consumed,
            entropy_before=entropy_before,
            entropy_after=state.memory_entropy,
        )

    def _apply_entropy_distortion(
        self, raw: str, cog: AgentCognition, name: str, tick: int
    ) -> str:
        """Entropy-induced distortion: may scramble action choice or target."""
        logger.debug("entropy distortion applied agent=%s tick=%d", name, tick)
        # Inject a marker so the downstream parse knows this was distorted
        # The JSON parser will still try to extract valid JSON; if entropy is
        # high enough the action may degrade to a random fallback.
        distorted = raw + ' ENTROPY_DISTORTED'
        return distorted
