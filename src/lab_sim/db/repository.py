from __future__ import annotations

import json
from typing import Any

from lab_sim.db.connection import DBClient
from lab_sim.utils.types import EventRecord, MemoryRecord, TrustVector


class LedgerRepository:
    def __init__(self, db: DBClient) -> None:
        self.db = db

    def create_run(
        self, run_id: str, mode: str, seed: int, agents: int, ticks: int, model: str
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, mode, seed, agents, ticks, model)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (run_id, mode, seed, agents, ticks, model),
            )

    def append_event(self, event: EventRecord) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (
                    run_id, tick, agent, action, payload, outcome, reward, cost, verified, importance
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
                """,
                (
                    event.run_id,
                    event.tick,
                    event.agent,
                    event.action,
                    json.dumps(event.payload),
                    event.outcome,
                    event.reward,
                    event.cost,
                    event.verified,
                    event.importance,
                ),
            )

    def append_memory(
        self,
        run_id: str,
        agent: str,
        tick: int,
        memory: MemoryRecord,
        embedding: list[float],
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories (
                  run_id, agent, tick, text, embedding, importance, verified, source_agent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    agent,
                    tick,
                    memory.text,
                    embedding,
                    memory.importance,
                    memory.verified,
                    memory.source_agent,
                ),
            )

    def upsert_trust(
        self, run_id: str, from_agent: str, to_agent: str, trust: TrustVector
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trust_edges (
                  run_id, from_agent, to_agent, truthfulness, consistency,
                  benevolence, competence, betrayal_count, alliance_strength, updated_tick
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, from_agent, to_agent)
                DO UPDATE SET
                  truthfulness = EXCLUDED.truthfulness,
                  consistency = EXCLUDED.consistency,
                  benevolence = EXCLUDED.benevolence,
                  competence = EXCLUDED.competence,
                  betrayal_count = EXCLUDED.betrayal_count,
                  alliance_strength = EXCLUDED.alliance_strength,
                  updated_tick = EXCLUDED.updated_tick
                """,
                (
                    run_id,
                    from_agent,
                    to_agent,
                    trust.truthfulness,
                    trust.consistency,
                    trust.benevolence,
                    trust.competence,
                    trust.betrayal_count,
                    trust.alliance_strength,
                    trust.updated_tick,
                ),
            )

    def get_trust(self, run_id: str, from_agent: str, to_agent: str) -> TrustVector:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT truthfulness, consistency, benevolence, competence,
                       betrayal_count, alliance_strength, updated_tick
                FROM trust_edges
                WHERE run_id = %s AND from_agent = %s AND to_agent = %s
                """,
                (run_id, from_agent, to_agent),
            )
            row = cur.fetchone()
            if row is None:
                return TrustVector()
            return TrustVector(
                truthfulness=float(row[0]),
                consistency=float(row[1]),
                benevolence=float(row[2]),
                competence=float(row[3]),
                betrayal_count=int(row[4]),
                alliance_strength=float(row[5]),
                updated_tick=int(row[6]),
            )

    # ---- Alliance persistence ----

    def upsert_alliance(
        self,
        run_id: str,
        agent_a: str,
        agent_b: str,
        formed_tick: int,
        strength: float,
        dissolved_tick: int | None = None,
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alliances (run_id, agent_a, agent_b, formed_tick, strength, dissolved_tick)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, agent_a, agent_b)
                DO UPDATE SET
                  strength = EXCLUDED.strength,
                  dissolved_tick = EXCLUDED.dissolved_tick
                """,
                (run_id, agent_a, agent_b, formed_tick, strength, dissolved_tick),
            )

    def get_alliances_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT agent_a, agent_b, formed_tick, dissolved_tick, strength
                FROM alliances
                WHERE run_id = %s
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "agent_a": str(r[0]),
                "agent_b": str(r[1]),
                "formed_tick": int(r[2]),
                "dissolved_tick": int(r[3]) if r[3] is not None else None,
                "strength": float(r[4]),
            }
            for r in rows
        ]

    # ---- Rumor persistence ----

    def append_rumor(
        self,
        run_id: str,
        claim_id: str,
        spreader: str,
        receiver: str,
        tick: int,
        mutation_degree: float,
        believed: bool,
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rumors (run_id, claim_id, spreader, receiver, tick, mutation_degree, believed)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, claim_id, spreader, receiver, tick, mutation_degree, believed),
            )

    def get_rumors_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT claim_id, spreader, receiver, tick, mutation_degree, believed
                FROM rumors
                WHERE run_id = %s
                ORDER BY tick
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "claim_id": str(r[0]),
                "spreader": str(r[1]),
                "receiver": str(r[2]),
                "tick": int(r[3]),
                "mutation_degree": float(r[4]),
                "believed": bool(r[5]),
            }
            for r in rows
        ]

    # ---- Existing query methods ----

    def append_state_snapshot(
        self,
        run_id: str,
        tick: int,
        agent: str,
        compute_budget: float,
        memory_entropy: float,
        reputation_score: float,
        tool_access_flags: dict,
        irreversible_damage: float,
        decision_latency_ms: float,
        used_heuristic: bool,
        fallback_reason: str,
    ) -> None:
        """Write a per-tick ecological state snapshot."""
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_state_snapshots (
                    run_id, tick, agent, compute_budget, memory_entropy,
                    reputation_score, tool_access_flags, irreversible_damage,
                    decision_latency_ms, used_heuristic, fallback_reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    tick,
                    agent,
                    compute_budget,
                    memory_entropy,
                    reputation_score,
                    json.dumps(tool_access_flags),
                    irreversible_damage,
                    decision_latency_ms,
                    used_heuristic,
                    fallback_reason or "",
                ),
            )

    def get_snapshots_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return all per-tick ecological state snapshots for a run."""
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT tick, agent, compute_budget, memory_entropy, reputation_score,
                       tool_access_flags, irreversible_damage, decision_latency_ms,
                       used_heuristic, fallback_reason
                FROM agent_state_snapshots
                WHERE run_id = %s
                ORDER BY tick, agent
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "tick": int(r[0]),
                "agent": str(r[1]),
                "compute_budget": float(r[2]),
                "memory_entropy": float(r[3]),
                "reputation_score": float(r[4]),
                "tool_access_flags": r[5] if isinstance(r[5], dict) else {},
                "irreversible_damage": float(r[6]),
                "decision_latency_ms": float(r[7]),
                "used_heuristic": bool(r[8]),
                "fallback_reason": str(r[9] or ""),
            }
            for r in rows
        ]

    # ---- Existing query methods ----

    def get_events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT tick, agent, action, payload, outcome, reward, cost, verified, importance, timestamp
                FROM events
                WHERE run_id = %s
                ORDER BY id
                """,
                (run_id,),
            )
            rows = cur.fetchall()

        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "tick": int(row[0]),
                    "agent": str(row[1]),
                    "action": str(row[2]),
                    "payload": row[3],
                    "outcome": str(row[4]),
                    "reward": float(row[5]),
                    "cost": float(row[6]),
                    "verified": bool(row[7]),
                    "importance": float(row[8]),
                    "timestamp": row[9].isoformat(),
                }
            )
        return output

    def get_trust_edges_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT from_agent, to_agent, truthfulness, consistency, benevolence,
                       competence, betrayal_count, alliance_strength, updated_tick
                FROM trust_edges
                WHERE run_id = %s
                """,
                (run_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "from_agent": str(r[0]),
                "to_agent": str(r[1]),
                "truthfulness": float(r[2]),
                "consistency": float(r[3]),
                "benevolence": float(r[4]),
                "competence": float(r[5]),
                "betrayal_count": int(r[6]),
                "alliance_strength": float(r[7]),
                "updated_tick": int(r[8]),
            }
            for r in rows
        ]
