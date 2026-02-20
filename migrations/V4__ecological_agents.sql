-- EcologicalAgents: computational survival constraints schema

-- Add compute cost and entropy tracking to events table
ALTER TABLE events
  ADD COLUMN IF NOT EXISTS compute_consumed  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS entropy_before    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS entropy_after     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS latency_ms        DOUBLE PRECISION NOT NULL DEFAULT 0.0;

-- Per-tick agent state snapshots for ecological survival tracking
CREATE TABLE IF NOT EXISTS agent_state_snapshots (
  id                    BIGSERIAL PRIMARY KEY,
  run_id                TEXT NOT NULL REFERENCES runs(run_id),
  tick                  INTEGER NOT NULL,
  agent                 TEXT NOT NULL,
  compute_budget        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  memory_entropy        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  reputation_score      DOUBLE PRECISION NOT NULL DEFAULT 0.5,
  tool_access_flags     JSONB NOT NULL DEFAULT '{}'::jsonb,
  irreversible_damage   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  decision_latency_ms   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  used_heuristic        BOOLEAN NOT NULL DEFAULT FALSE,
  fallback_reason       TEXT,
  timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run_tick  ON agent_state_snapshots(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_snapshots_run_agent ON agent_state_snapshots(run_id, agent);
