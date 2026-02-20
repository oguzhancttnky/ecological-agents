CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  seed INTEGER NOT NULL,
  agents INTEGER NOT NULL,
  ticks INTEGER NOT NULL,
  model TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  tick INTEGER NOT NULL,
  agent TEXT NOT NULL,
  action TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  outcome TEXT NOT NULL,
  reward DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  cost DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  verified BOOLEAN NOT NULL DEFAULT FALSE,
  importance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_run_tick ON events(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_events_run_agent ON events(run_id, agent);

CREATE TABLE IF NOT EXISTS memories (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  agent TEXT NOT NULL,
  tick INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(768) NOT NULL,
  importance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  verified BOOLEAN NOT NULL DEFAULT FALSE,
  source_agent TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_run_agent_tick ON memories(run_id, agent, tick DESC);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS trust_edges (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  from_agent TEXT NOT NULL,
  to_agent TEXT NOT NULL,
  truthfulness DOUBLE PRECISION NOT NULL,
  consistency DOUBLE PRECISION NOT NULL,
  benevolence DOUBLE PRECISION NOT NULL,
  competence DOUBLE PRECISION NOT NULL,
  updated_tick INTEGER NOT NULL,
  PRIMARY KEY (run_id, from_agent, to_agent)
);

CREATE INDEX IF NOT EXISTS idx_trust_edges_from ON trust_edges(run_id, from_agent);
