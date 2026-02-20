from __future__ import annotations

from dataclasses import dataclass
import logging

from lab_sim.db.connection import DBClient
from lab_sim.db.repository import LedgerRepository
from lab_sim.llm.ollama_adapter import OllamaAdapter
from lab_sim.utils.types import MemoryRecord


@dataclass
class RetrievedMemory:
    text: str
    tick: int
    source_agent: str
    verified: bool
    importance: float
    trust_score: float
    similarity: float
    final_score: float


class VectorMemoryService:
    def __init__(
        self, db: DBClient, repo: LedgerRepository, ollama: OllamaAdapter, mode: str
    ) -> None:
        self.logger = logging.getLogger("lab_sim.memory")
        self.db = db
        self.repo = repo
        self.ollama = ollama
        self.mode = mode
        self._mode_set = set(p.strip() for p in mode.split("+") if p.strip())
        self._write_count = 0
        self._retrieve_count = 0

    def write_memory(
        self,
        run_id: str,
        agent: str,
        tick: int,
        text: str,
        importance: float,
        verified: bool,
        source_agent: str,
    ) -> None:
        if "no_memory" in self._mode_set:
            return
        try:
            emb = self.ollama.embed(text)
        except Exception as exc:
            self.logger.warning(
                "Memory write skipped due to embed failure: run_id=%s agent=%s tick=%d error=%s",
                run_id,
                agent,
                tick,
                exc.__class__.__name__,
            )
            return
        self.repo.append_memory(
            run_id=run_id,
            agent=agent,
            tick=tick,
            memory=MemoryRecord(
                text=text,
                importance=importance,
                verified=verified,
                source_agent=source_agent,
                tick=tick,
            ),
            embedding=emb,
        )
        self._write_count += 1
        if self._write_count % 100 == 0:
            self.logger.info(
                "Memory write progress: mode=%s writes=%d latest_tick=%d latest_agent=%s",
                self.mode,
                self._write_count,
                tick,
                agent,
            )

    def retrieve(
        self,
        run_id: str,
        agent: str,
        query: str,
        tick: int,
        limit: int = 10,
    ) -> list[RetrievedMemory]:
        if "no_memory" in self._mode_set:
            return []

        try:
            query_emb = self.ollama.embed(query)
        except Exception as exc:
            self.logger.warning(
                "Memory retrieval degraded to empty due to embed failure: run_id=%s agent=%s tick=%d error=%s",
                run_id,
                agent,
                tick,
                exc.__class__.__name__,
            )
            return []
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT
                  m.text,
                  m.tick,
                  m.source_agent,
                  m.verified,
                  m.importance,
                  COALESCE(
                    te.truthfulness * 0.35
                    + te.consistency * 0.25
                    + te.benevolence * 0.20
                    + te.competence * 0.20,
                    0.5
                  ) AS trust_score,
                  (1 - (m.embedding <=> %s::vector)) AS similarity
                FROM memories m
                LEFT JOIN trust_edges te
                  ON te.run_id = m.run_id
                 AND te.from_agent = %s
                 AND te.to_agent = m.source_agent
                WHERE m.run_id = %s
                  AND m.agent = %s
                  AND m.tick <= %s
                ORDER BY m.embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (query_emb, agent, run_id, agent, tick, query_emb, limit * 3),
            )
            rows = cur.fetchall()

        reranked: list[RetrievedMemory] = []
        for row in rows:
            memory_tick = int(row[1])
            recency = 1.0 / (1.0 + max(0, tick - memory_tick))
            verified_bonus = 1.0 if bool(row[3]) else 0.6
            trust_score = float(row[5])
            if "no_trust" in self._mode_set:
                trust_score = 0.5
            final_score = (
                float(row[6]) * 0.4
                + float(row[4]) * 0.25
                + verified_bonus * 0.15
                + trust_score * 0.1
                + recency * 0.1
            )
            reranked.append(
                RetrievedMemory(
                    text=str(row[0]),
                    tick=memory_tick,
                    source_agent=str(row[2]),
                    verified=bool(row[3]),
                    importance=float(row[4]),
                    trust_score=trust_score,
                    similarity=float(row[6]),
                    final_score=final_score,
                )
            )
        reranked.sort(key=lambda m: m.final_score, reverse=True)
        self._retrieve_count += 1
        if self._retrieve_count % 50 == 0:
            self.logger.info(
                "Memory retrieval progress: mode=%s retrievals=%d run_id=%s agent=%s tick=%d returned=%d",
                self.mode,
                self._retrieve_count,
                run_id,
                agent,
                tick,
                min(limit, len(reranked)),
            )
        return reranked[:limit]
