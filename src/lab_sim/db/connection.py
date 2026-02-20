from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PgConnection

from lab_sim.config.settings import DBSettings


class DBClient:
    def __init__(self, settings: DBSettings) -> None:
        self._settings = settings
        self._conn: PgConnection | None = None

    def connect(self) -> None:
        if self._conn is None:
            self._conn = psycopg2.connect(self._settings.dsn)
            self._conn.autocommit = False
            register_vector(self._conn)

    @property
    def conn(self) -> PgConnection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    @contextmanager
    def cursor(self) -> Iterator:
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
