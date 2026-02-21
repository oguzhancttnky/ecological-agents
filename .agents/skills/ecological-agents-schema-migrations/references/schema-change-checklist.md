# Schema Change Checklist

## Before Writing SQL

1. Identify affected repository methods in `src/lab_sim/db/repository.py`.
2. Identify affected metrics in `src/lab_sim/metrics/engine.py`.
3. Decide whether data backfill is needed for historical runs.

## SQL Authoring

1. Use a new `V{n}__*.sql` file in `migrations/`.
2. Keep operations idempotent where practical.
3. Add indexes for new lookup/join paths.
4. Preserve old columns temporarily if application code still reads them.

## Verification

1. Apply with Flyway.
2. Run a short simulation batch.
3. Confirm `metrics.json` and `metrics.csv` still populate expected fields.
4. Confirm no repository query errors in logs.
