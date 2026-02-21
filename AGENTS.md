# AGENTS.md

## Scope

This file defines project-specific instructions for AI coding agents working in this repository.

Project: `EcologicalAgents`  
Primary stack: Python 3.11, PostgreSQL + pgvector, Flyway, Docker Compose, Ollama.

## Repository Map

- `src/lab_sim/`: simulation, cognition, trust, memory, DB, metrics
- `migrations/`: Flyway SQL migrations (`V{n}__description.sql`)
- `flyway/`: Flyway configuration
- `.env` / `.env.example`: runtime and experiment settings
- `outputs/`: run artifacts and aggregate metrics
- `.agents/skills/`: project-local Codex skills

## Default Workflow

1. Read `README.md`, `.env.example`, and relevant module files before changing behavior.
2. Prefer minimal, targeted edits that preserve existing experiment comparability.
3. Keep experiment plans reproducible (explicit `EXPERIMENT_MODES` and `EXPERIMENT_SEEDS`).
4. Validate changes with the smallest representative run before broad sweeps.
5. Report exact commands executed and output locations produced.

## Runtime Commands

```bash
cp .env.example .env
docker compose up -d postgres
docker compose run --rm flyway
docker compose run --rm app
```

For attached logs:

```bash
docker compose up --build
```

## Skill Usage

Use these project skills when task intent matches:

- `$ecological-agents-experiment-runner`
Description: execute/troubleshoot simulation batches, plan runs, and validate outputs.
- `$ecological-agents-metrics-analysis`
Description: aggregate/compare metrics across modes and seeds from `outputs/`.
- `$ecological-agents-schema-migrations`
Description: create/apply migration files and verify repository/metrics compatibility.

If a request clearly matches one of these domains, invoke the corresponding skill for that turn.

## Safety Rules

- Do not rewrite or delete historical run artifacts unless explicitly requested.
- Do not edit old migration versions already committed; add a new migration file.
- Avoid silently changing default environment parameters that impact benchmark comparability.
- Treat causal claims from ablations cautiously unless controls are matched and documented.

## Code Quality

- Preserve ASCII unless file content already requires otherwise.
- Keep comments concise and only where they add non-obvious context.
- Prefer `rg` for file/text search.
- Prefer non-interactive git commands.

## Output Expectations

When experiments are run or analyzed, include:

1. run plan (`modes`, `seeds`, total runs),
2. output paths (`outputs/<run_id>/...` and `outputs/metrics.csv`),
3. key metric deltas with explicit direction (higher/lower is better).
