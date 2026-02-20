CREATE OR REPLACE FUNCTION prevent_mutation()
RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'append-only table: %', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS runs_no_update ON runs;
DROP TRIGGER IF EXISTS runs_no_delete ON runs;
DROP TRIGGER IF EXISTS events_no_update ON events;
DROP TRIGGER IF EXISTS events_no_delete ON events;
DROP TRIGGER IF EXISTS memories_no_update ON memories;
DROP TRIGGER IF EXISTS memories_no_delete ON memories;

CREATE TRIGGER runs_no_update
BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER runs_no_delete
BEFORE DELETE ON runs
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER events_no_update
BEFORE UPDATE ON events
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER events_no_delete
BEFORE DELETE ON events
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER memories_no_update
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();

CREATE TRIGGER memories_no_delete
BEFORE DELETE ON memories
FOR EACH ROW EXECUTE FUNCTION prevent_mutation();
