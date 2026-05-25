"""
Residency Store
---------------
SQLite-backed persistence for per-(course, lane, batch) EWMA residency
accumulators.  The database is read once at controller startup to seed
ResidencyStats, then periodically overwritten with the current in-memory
state so that learned parameters survive restarts.

Schema
~~~~~~
    residency_strata (
        course_id  TEXT,
        lane_name  TEXT,
        batch      INTEGER,   -- 0 = interactive, 1 = batch
        ewma_mean  REAL,      -- EWMA mean residency fraction
        ewma_var   REAL,      -- EWMA variance
        n          INTEGER,   -- observation count
        updated_at TEXT,      -- ISO-8601 wall-clock timestamp
        PRIMARY KEY (course_id, lane_name, batch)
    )

Thread safety
~~~~~~~~~~~~~
Each public method opens its own short-lived connection, so the store is
safe to call from any thread without external locking.  WAL mode is
enabled to allow concurrent readers.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Row type: (course_id, lane_name, batch, ewma_mean, ewma_var, n)
StratumRow = tuple[str, str, bool, float, float, int]

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS residency_strata (
        course_id  TEXT    NOT NULL,
        lane_name  TEXT    NOT NULL,
        batch      INTEGER NOT NULL,
        ewma_mean  REAL    NOT NULL,
        ewma_var   REAL    NOT NULL,
        n          INTEGER NOT NULL,
        updated_at TEXT    NOT NULL,
        PRIMARY KEY (course_id, lane_name, batch)
    )
"""

_UPSERT = """
    INSERT OR REPLACE INTO residency_strata
        (course_id, lane_name, batch, ewma_mean, ewma_var, n, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_ALL = """
    SELECT course_id, lane_name, batch, ewma_mean, ewma_var, n
    FROM residency_strata
"""


class ResidencyStore:
    """
    Read/write EWMA residency state to a SQLite database.

    Usage
    -----
        store = ResidencyStore(Path("/var/lib/lane-scheduler/residency.db"))

        # At startup — seed ResidencyStats:
        for course_id, lane_name, batch, mean, var, n in store.load():
            residency_stats.seed(course_id, lane_name, batch, mean, var, n)

        # Periodically — flush in-memory state:
        store.save(residency_stats.dump())
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[StratumRow]:
        """
        Return all stored strata as (course_id, lane_name, batch, mean, var, n).
        Returns an empty list if the database is empty or does not yet exist.
        """
        with self._connect() as conn:
            rows = conn.execute(_SELECT_ALL).fetchall()
        return [
            (course_id, lane_name, bool(batch), mean, var, n)
            for course_id, lane_name, batch, mean, var, n in rows
        ]

    def save(self, rows: list[StratumRow]) -> None:
        """
        Upsert all rows into the database.

        Rows with n=0 are skipped (nothing to persist).
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        params = [
            (course_id, lane_name, int(batch), mean, var, n, now)
            for course_id, lane_name, batch, mean, var, n in rows
            if n > 0
        ]
        if not params:
            return
        with self._connect() as conn:
            conn.executemany(_UPSERT, params)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
        logger.debug("Residency DB schema ready at %s", self.path)
