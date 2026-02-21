---
name: ecological-agents-schema-migrations
description: Maintain and apply EcologicalAgents PostgreSQL schema migrations with Flyway and pgvector compatibility. Use when Codex needs to add or modify tables used by simulation, memory, trust, or metrics subsystems, and keep migration history consistent.
---

# Ecological Agents Schema Migrations

## Overview

Create safe, forward-only migration files and verify compatibility with repository queries and metrics computation.

## Workflow

1. Read `migrations/` in numeric order and inspect `src/lab_sim/db/repository.py` for query expectations.
2. Add a new migration file with the next version number (`V{n}__description.sql`).
3. Keep migrations additive and backward-compatible where possible.
4. Apply migrations through Flyway.
5. Verify app startup and metrics generation still succeed.

## Migration Rules

- Never edit already-applied migration files in shared environments.
- Prefer additive changes over destructive rewrites.
- For column renames/removals, stage with compatibility windows.
- Maintain pgvector extension assumptions used by memory retrieval.
- Ensure indexes align with read paths in repository methods.

## Execute

```bash
docker compose run --rm flyway
```

For clean local reset only when explicitly requested:

```bash
docker compose down -v
```

## Validation Focus

- `src/lab_sim/db/repository.py` query shape and selected columns.
- `src/lab_sim/metrics/engine.py` metrics dependencies (events, trust edges, rumors, alliances, snapshots).
- Run artifact writing path in `src/lab_sim/experiments/runner.py`.

## References

Use:

- `references/schema-change-checklist.md` for change impact checks.
