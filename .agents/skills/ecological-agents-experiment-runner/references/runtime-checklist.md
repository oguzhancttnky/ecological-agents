# Runtime Checklist

## Preflight

1. Ensure `.env` exists and contains DB, Ollama, simulation, and experiment plan keys.
2. Ensure Docker daemon is running.
3. Ensure Ollama host is reachable and the configured `LLM_MODEL` plus `EMBEDDING_MODEL` are available.
4. Ensure PostgreSQL port in `.env` matches `docker-compose.yml` mapping.

## Run Plan

1. Parse `EXPERIMENT_MODES` as comma-separated values.
2. Parse `EXPERIMENT_SEEDS` as comma-separated integers.
3. Compute expected runs as `len(modes) * len(seeds)`.

## Execution

1. Start or verify `postgres` service health.
2. Run `flyway` migrations.
3. Run `app` container to execute `python -m lab_sim.experiments.run_experiments`.

## Post-Run Validation

1. Confirm `outputs/metrics.csv` exists and has one row per run.
2. Confirm each run directory contains:
   - `events.json`
   - `trust_edges.json`
   - `alliances.json`
   - `rumors.json`
   - `metrics.json`
3. Spot-check logs for failed ticks, repeated timeouts, or missing metrics keys.
