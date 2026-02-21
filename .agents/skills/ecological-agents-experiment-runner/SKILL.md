---
name: ecological-agents-experiment-runner
description: Run and troubleshoot EcologicalAgents simulation batches with reproducible mode/seed plans using Docker Compose and environment variables. Use when Codex needs to execute experiments, adjust run plans, verify Ollama/Postgres/Flyway readiness, or generate run artifacts under outputs/.
---

# Ecological Agents Experiment Runner

## Overview

Run experiment batches safely and reproducibly, then verify that expected artifacts are created per run.

## Workflow

1. Read `README.md`, `.env`, `.env.example`, `docker-compose.yml`, and `src/lab_sim/experiments/run_experiments.py` before changing execution settings.
2. Validate run plan in `EXPERIMENT_MODES` and `EXPERIMENT_SEEDS` and estimate total run count (`modes x seeds`).
3. Ensure dependencies are reachable.
4. Launch batch execution.
5. Confirm run outputs and top-level `metrics.csv` were written.

## Execute Batch

Use these commands in order:

```bash
cp .env.example .env
```

```bash
docker compose up -d postgres
```

```bash
docker compose run --rm flyway
```

```bash
docker compose run --rm app
```

For full stack startup with attached logs:

```bash
docker compose up --build
```

## Verify Output

Check expected artifacts:

```bash
ls -la outputs
```

Per run, verify `events.json`, `trust_edges.json`, `alliances.json`, `rumors.json`, and `metrics.json` exist.

## Safety Rules

- Keep ablation comparisons fair by changing one factor at a time.
- Preserve exact mode and seed lists used for a report.
- Prefer appending new output directories over overwriting prior results.
- Do not silently change `.env` defaults without stating rationale.

## Troubleshooting

- If app fails at DB connection, verify `DB_HOST`, `DB_PORT`, and PostgreSQL container health.
- If migrations fail, inspect `flyway/flyway.conf` and `migrations/` ordering.
- If LLM calls stall, inspect `OLLAMA_HOST`, model availability, and timeout settings.
- If run throughput is poor, tune `MAX_CONCURRENT_LLM`, `TICK_REQUEST_TIMEOUT_S`, and `TICK_DEADLINE_S` conservatively.

## References

Read these only when needed:

- `references/runtime-checklist.md` for preflight and run validation.
