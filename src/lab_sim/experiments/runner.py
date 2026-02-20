from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from lab_sim.agents.cognition import AgentCognition
from lab_sim.config.settings import AppSettings
from lab_sim.db.connection import DBClient
from lab_sim.db.repository import LedgerRepository
from lab_sim.llm.ollama_adapter import OllamaAdapter
from lab_sim.memory.vector_memory import VectorMemoryService
from lab_sim.metrics.engine import MetricsEngine
from lab_sim.trust.system import TrustSystem
from lab_sim.world.simulator import WorldSimulator


@dataclass
class ExperimentSpec:
    mode: str
    seed: int


class ExperimentRunner:
    def __init__(self, settings: AppSettings) -> None:
        self.logger = logging.getLogger("lab_sim.runner")
        self.settings = settings
        self.db = DBClient(settings.db)
        self.db.connect()
        self.repo = LedgerRepository(self.db)
        self.ollama = OllamaAdapter(settings.ollama)
        self.metrics_engine = MetricsEngine(self.repo)

    def run_many(self, specs: Iterable[ExperimentSpec]) -> list[dict]:
        rows: list[dict] = []
        specs_list = list(specs)
        self.logger.info("Starting batch execution: run_count=%d", len(specs_list))
        batch_start = time.perf_counter()
        for idx, spec in enumerate(specs_list, start=1):
            self.logger.info(
                "Run queued: index=%d/%d mode=%s seed=%d",
                idx,
                len(specs_list),
                spec.mode,
                spec.seed,
            )
            row = self.run_one(spec.mode, spec.seed)
            rows.append(row)
        metrics_path = self.metrics_engine.write_metrics_csv(
            self.settings.output_dir, rows, filename="metrics.csv"
        )
        self.logger.info(
            "Batch completed in %.2fs. Aggregate metrics at %s",
            time.perf_counter() - batch_start,
            metrics_path,
        )
        return rows

    def run_one(self, mode: str, seed: int) -> dict:
        run_id = self._run_id(mode, seed)
        run_start = time.perf_counter()
        self.logger.info("Starting run: %s", run_id)
        self.repo.create_run(
            run_id=run_id,
            mode=mode,
            seed=seed,
            agents=self.settings.simulation.agent_count,
            ticks=self.settings.simulation.ticks,
            model=self.settings.ollama.llm_model,
        )
        memory = VectorMemoryService(self.db, self.repo, self.ollama, mode=mode)
        trust_system = TrustSystem(self.repo, mode=mode)

        def cognition_factory(name, state, rng):
            return AgentCognition(
                name=name,
                state=state,
                memory=memory,
                ollama=self.ollama,
                sim=self.settings.simulation,
                mode=mode,
                rng=rng,
            )

        sim = WorldSimulator(
            run_id=run_id,
            mode=mode,
            seed=seed,
            settings=self.settings,
            repo=self.repo,
            memory=memory,
            trust_system=trust_system,
            cognition_factory=cognition_factory,
        )
        sim.run()
        self.logger.info("Simulation completed for run: %s", run_id)
        run_metrics = self.metrics_engine.compute(run_id)
        self.logger.info("Metrics computed for run: %s -> %s", run_id, run_metrics)
        self._write_run_artifacts(run_id, run_metrics)
        self.logger.info(
            "Completed run: %s in %.2fs", run_id, time.perf_counter() - run_start
        )
        return {"run_id": run_id, "mode": mode, "seed": seed, **run_metrics}

    def close(self) -> None:
        self.db.close()

    def _write_run_artifacts(self, run_id: str, run_metrics: dict) -> None:
        run_dir = self.settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        events = self.repo.get_events_for_run(run_id)
        trust = self.repo.get_trust_edges_for_run(run_id)
        alliances = self.repo.get_alliances_for_run(run_id)
        rumors = self.repo.get_rumors_for_run(run_id)

        (run_dir / "events.json").write_text(
            json.dumps(events, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (run_dir / "trust_edges.json").write_text(
            json.dumps(trust, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (run_dir / "alliances.json").write_text(
            json.dumps(alliances, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (run_dir / "rumors.json").write_text(
            json.dumps(rumors, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(run_metrics, indent=2, ensure_ascii=True), encoding="utf-8"
        )

    def _run_id(self, mode: str, seed: int) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{ts}_{mode}_seed{seed}"


def parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def build_specs(modes: list[str], seeds: list[int]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for mode in modes:
        for seed in seeds:
            specs.append(ExperimentSpec(mode=mode, seed=seed))
    return specs
