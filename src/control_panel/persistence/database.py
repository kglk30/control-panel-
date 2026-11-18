"""SQLite connection management with WAL mode using aiosqlite."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import aiosqlite

from control_panel.common.logging import get_logger

__all__ = ["Database", "get_database", "close_database"]

log = get_logger(__name__)


class Database:
    """Async SQLite connection wrapper with WAL mode and transaction support."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Open connection, enable WAL mode and foreign keys, run migrations."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()

        # Run schema migrations
        from control_panel.persistence.migrations import run_migrations
        await run_migrations(self)
        log.info("database.initialized", path=str(self._db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.info("database.closed")

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager for an explicit database transaction.

        Commits on clean exit, rolls back on exception.
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized — call initialize() first")
        try:
            yield self._conn
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a single SQL statement and return the cursor."""
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        cursor = await self._conn.execute(sql, params)
        await self._conn.commit()
        return cursor

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[aiosqlite.Row]:
        """Fetch a single row, or None if no rows matched."""
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        async with self._conn.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        """Fetch all matching rows."""
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        async with self._conn.execute(sql, params) as cursor:
            return await cursor.fetchall()

    async def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a statement with multiple parameter sets."""
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        await self._conn.executemany(sql, params_list)
        await self._conn.commit()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_db: Optional[Database] = None


async def get_database(db_path: Optional[Path] = None) -> Database:
    """Get or initialize the global Database singleton."""
    global _db
    if _db is None:
        if db_path is None:
            from control_panel.common.config import get_app_dirs
            db_path = get_app_dirs().database_path
        _db = Database(db_path)
        await _db.initialize()
    return _db


async def close_database() -> None:
    """Close and clear the global database singleton."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
