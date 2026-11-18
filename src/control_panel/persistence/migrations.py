"""SQLite schema migrations for Control Panel.

All tables use CREATE TABLE IF NOT EXISTS so migrations are idempotent.
New columns in future versions will be added with ALTER TABLE.
"""

from __future__ import annotations

from control_panel.common.logging import get_logger

if False:  # TYPE_CHECKING
    from control_panel.persistence.database import Database

__all__ = ["run_migrations"]

log = get_logger(__name__)

_TABLES = [
    # ------------------------------------------------------------------
    # templates — reusable workflow strategies
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS templates (
        id                TEXT NOT NULL,
        version           INTEGER NOT NULL,
        name              TEXT NOT NULL,
        category          TEXT NOT NULL,
        description       TEXT,
        source            TEXT NOT NULL,
        workflow_json     TEXT NOT NULL,
        requirements_json TEXT,
        performance_json  TEXT,
        usage_count       INTEGER DEFAULT 0,
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        PRIMARY KEY (id, version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tmpl_category ON templates(category)",

    # ------------------------------------------------------------------
    # experiences — historical task execution records (V13 expanded)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS experiences (
        id                        TEXT PRIMARY KEY,
        task_id                   TEXT NOT NULL,
        category                  TEXT NOT NULL,
        raw_input                 TEXT NOT NULL,
        task_spec_json            TEXT NOT NULL,
        plan_json                 TEXT NOT NULL,
        inference_json            TEXT NOT NULL,
        execution_json            TEXT NOT NULL,
        resource_context_json     TEXT NOT NULL,
        outcome_json              TEXT NOT NULL,
        strategy_json             TEXT,
        success                   INTEGER NOT NULL,
        total_runtime_ms          INTEGER,
        backend_used              TEXT,
        device_used               TEXT,
        strategy_type             TEXT,
        worker_count              INTEGER,
        parallelizable_fraction   REAL,
        is_archived               INTEGER DEFAULT 0,
        created_at                TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_exp_category ON experiences(category)",
    "CREATE INDEX IF NOT EXISTS idx_exp_success ON experiences(success)",
    "CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_exp_backend ON experiences(backend_used)",
    "CREATE INDEX IF NOT EXISTS idx_exp_archived ON experiences(is_archived)",

    # ------------------------------------------------------------------
    # rollback_log — undo log for mutating filesystem operations
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS rollback_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id             TEXT NOT NULL,
        worker_id           TEXT NOT NULL,
        side_effect_type    TEXT NOT NULL,
        params_json         TEXT NOT NULL,
        undo_params_json    TEXT NOT NULL,
        executed_at         TEXT NOT NULL,
        rolled_back         INTEGER DEFAULT 0,
        rolled_back_at      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rollback_task ON rollback_log(task_id)",

    # ------------------------------------------------------------------
    # accelerator_probes — AcceleratorProbe results cache (V13)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS accelerator_probes (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        accelerator_type    TEXT NOT NULL,
        available           INTEGER NOT NULL,
        device_name         TEXT,
        provider            TEXT,
        driver_version      TEXT,
        provider_loadable   INTEGER,
        model_compatible    INTEGER,
        probe_inference_ok  INTEGER,
        failure_reason      TEXT,
        probed_at           TEXT NOT NULL
    )
    """,

    # ------------------------------------------------------------------
    # experience_clusters — clustered experience memory (V15)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS experience_clusters (
        cluster_id                TEXT PRIMARY KEY,
        category                  TEXT NOT NULL,
        name                      TEXT NOT NULL,
        representative_json       TEXT NOT NULL,
        strategies_json           TEXT NOT NULL,
        statistics_json           TEXT NOT NULL,
        failure_patterns_json     TEXT,
        exceptional_ids_json      TEXT,
        total_observations        INTEGER DEFAULT 0,
        overall_success_rate      REAL DEFAULT 0.0,
        best_strategy_type        TEXT,
        created_at                TEXT NOT NULL,
        updated_at                TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cluster_category ON experience_clusters(category)",

    # ------------------------------------------------------------------
    # experience_features — feature vectors for clustering (V15)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS experience_features (
        experience_id             TEXT PRIMARY KEY,
        cluster_id                TEXT,
        task_features_json        TEXT NOT NULL,
        workflow_features_json    TEXT NOT NULL,
        compute_features_json     TEXT NOT NULL,
        performance_features_json TEXT NOT NULL,
        is_exceptional            INTEGER DEFAULT 0,
        is_archived               INTEGER DEFAULT 0,
        created_at                TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_features_cluster ON experience_features(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_features_archived ON experience_features(is_archived)",
]


async def run_migrations(db: "Database") -> None:
    """Create all tables and indexes if they don't exist.

    Idempotent — safe to call on every startup.
    """
    async with db.transaction() as conn:
        for statement in _TABLES:
            stmt = statement.strip()
            if stmt:
                await conn.execute(stmt)
    log.info("migrations.complete", tables=6)
