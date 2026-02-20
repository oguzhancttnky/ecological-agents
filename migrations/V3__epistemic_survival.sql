-- Epistemic survival: alliance tracking and rumor propagation

CREATE TABLE IF NOT EXISTS alliances (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  agent_a TEXT NOT NULL,
  agent_b TEXT NOT NULL,
  formed_tick INTEGER NOT NULL,
  dissolved_tick INTEGER,
  strength DOUBLE PRECISION NOT NULL DEFAULT 0.5,
  PRIMARY KEY (run_id, agent_a, agent_b)
);

CREATE INDEX IF NOT EXISTS idx_alliances_run ON alliances(run_id);

CREATE TABLE IF NOT EXISTS rumors (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  claim_id TEXT NOT NULL,
  spreader TEXT NOT NULL,
  receiver TEXT NOT NULL,
  tick INTEGER NOT NULL,
  mutation_degree DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  believed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_rumors_run_tick ON rumors(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_rumors_claim ON rumors(run_id, claim_id);

-- Add betrayal_count and alliance_strength to trust_edges
ALTER TABLE trust_edges
  ADD COLUMN IF NOT EXISTS betrayal_count INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS alliance_strength DOUBLE PRECISION NOT NULL DEFAULT 0.0;
