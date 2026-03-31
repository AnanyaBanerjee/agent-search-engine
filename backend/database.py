"""
Dual-backend store for agent cards + numpy vector index.

- When DATABASE_URL env var is set (Railway/production) → PostgreSQL
- Otherwise → SQLite (local development, zero setup)

The public API is identical in both modes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from models import AgentCard

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
_USE_POSTGRES = DATABASE_URL is not None

# SQLite fallback path (local dev only)
_SQLITE_PATH = Path(__file__).parent / "agents.db"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def _pg_conn():
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _sqlite_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _conn():
    return _pg_conn() if _USE_POSTGRES else _sqlite_conn()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS agents (
    id                TEXT PRIMARY KEY,
    card_json         TEXT        NOT NULL,
    embedding         BYTEA       NOT NULL,
    registered        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status            TEXT        NOT NULL DEFAULT 'unknown',
    last_checked      TIMESTAMPTZ,
    last_seen_online  TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS agent_versions (
    id          SERIAL PRIMARY KEY,
    agent_id    TEXT        NOT NULL,
    version_num INTEGER     NOT NULL,
    card_json   TEXT        NOT NULL,
    diff_json   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_versions_agent_id ON agent_versions (agent_id);
CREATE TABLE IF NOT EXISTS search_logs (
    id            SERIAL PRIMARY KEY,
    query         TEXT        NOT NULL,
    results_count INTEGER     NOT NULL DEFAULT 0,
    tags_json     TEXT        NOT NULL DEFAULT '[]',
    searched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS agent_clicks (
    id         SERIAL PRIMARY KEY,
    agent_id   TEXT        NOT NULL,
    query_text TEXT,
    clicked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_clicks_agent_id ON agent_clicks (agent_id);
CREATE TABLE IF NOT EXISTS agent_impressions (
    id        SERIAL PRIMARY KEY,
    agent_id  TEXT        NOT NULL,
    query     TEXT        NOT NULL DEFAULT '',
    shown_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_impressions_agent_id ON agent_impressions (agent_id);
CREATE TABLE IF NOT EXISTS agent_task_centroids (
    agent_id    TEXT    PRIMARY KEY,
    centroid    BYTEA   NOT NULL,
    click_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_reviews (
    id          SERIAL PRIMARY KEY,
    agent_id    TEXT        NOT NULL,
    reviewer_id TEXT        NOT NULL,
    score       INTEGER     NOT NULL CHECK (score BETWEEN 1 AND 5),
    comment     TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (agent_id, reviewer_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_reviews_agent_id ON agent_reviews (agent_id);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS agents (
    id                TEXT PRIMARY KEY,
    card_json         TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    registered        TEXT DEFAULT (datetime('now')),
    status            TEXT NOT NULL DEFAULT 'unknown',
    last_checked      TEXT,
    last_seen_online  TEXT
);
CREATE TABLE IF NOT EXISTS agent_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    version_num INTEGER NOT NULL,
    card_json   TEXT    NOT NULL,
    diff_json   TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_versions_agent_id ON agent_versions (agent_id);
CREATE TABLE IF NOT EXISTS search_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query         TEXT    NOT NULL,
    results_count INTEGER NOT NULL DEFAULT 0,
    tags_json     TEXT    NOT NULL DEFAULT '[]',
    searched_at   TEXT    DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS agent_clicks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT    NOT NULL,
    query_text TEXT,
    clicked_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_clicks_agent_id ON agent_clicks (agent_id);
CREATE TABLE IF NOT EXISTS agent_impressions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT    NOT NULL,
    query    TEXT    NOT NULL DEFAULT '',
    shown_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_impressions_agent_id ON agent_impressions (agent_id);
CREATE TABLE IF NOT EXISTS agent_task_centroids (
    agent_id    TEXT    PRIMARY KEY,
    centroid    BLOB    NOT NULL,
    click_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    reviewer_id TEXT    NOT NULL,
    score       INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
    comment     TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE (agent_id, reviewer_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_reviews_agent_id ON agent_reviews (agent_id);
"""


def _migrate_agents_columns(conn, use_postgres: bool) -> None:
    """Add new columns to existing tables for schema upgrades."""
    agents_cols = [
        ("status",           "TEXT NOT NULL DEFAULT 'unknown'"),
        ("last_checked",     "TIMESTAMPTZ" if use_postgres else "TEXT"),
        ("last_seen_online", "TIMESTAMPTZ" if use_postgres else "TEXT"),
    ]
    clicks_cols = [
        ("query_text", "TEXT"),
    ]
    if use_postgres:
        cur = conn.cursor()
        for col, typedef in agents_cols:
            cur.execute(f"ALTER TABLE agents ADD COLUMN IF NOT EXISTS {col} {typedef}")
        for col, typedef in clicks_cols:
            cur.execute(f"ALTER TABLE agent_clicks ADD COLUMN IF NOT EXISTS {col} {typedef}")
    else:
        existing_agents = {row["name"] for row in conn.execute("PRAGMA table_info(agents)").fetchall()}
        for col, typedef in agents_cols:
            if col not in existing_agents:
                conn.execute(f"ALTER TABLE agents ADD COLUMN {col} {typedef}")
        try:
            existing_clicks = {row["name"] for row in conn.execute("PRAGMA table_info(agent_clicks)").fetchall()}
            for col, typedef in clicks_cols:
                if col not in existing_clicks:
                    conn.execute(f"ALTER TABLE agent_clicks ADD COLUMN {col} {typedef}")
        except Exception:
            pass


def init_db() -> None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            for statement in _SCHEMA_PG.strip().split(";"):
                s = statement.strip()
                if s:
                    cur.execute(s)
            _migrate_agents_columns(conn, use_postgres=True)
    else:
        with _sqlite_conn() as conn:
            conn.executescript(_SCHEMA_SQLITE)
            _migrate_agents_columns(conn, use_postgres=False)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert_agent(agent_id: str, card: AgentCard, embedding: np.ndarray) -> str:
    blob = embedding.astype(np.float32).tobytes()
    card_json = card.model_dump_json()

    if _USE_POSTGRES:
        import psycopg2
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO agents (id, card_json, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    card_json  = EXCLUDED.card_json,
                    embedding  = EXCLUDED.embedding,
                    registered = NOW()
                """,
                (agent_id, card_json, psycopg2.Binary(blob)),
            )
    else:
        with _sqlite_conn() as conn:
            conn.execute(
                """
                INSERT INTO agents (id, card_json, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    card_json  = excluded.card_json,
                    embedding  = excluded.embedding,
                    registered = datetime('now')
                """,
                (agent_id, card_json, blob),
            )
    return agent_id


def get_agent(agent_id: str) -> AgentCard | None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT card_json FROM agents WHERE id = %s", (agent_id,))
            row = cur.fetchone()
    else:
        with _sqlite_conn() as conn:
            row = conn.execute(
                "SELECT card_json FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()

    if row is None:
        return None
    return AgentCard.model_validate_json(row[0])


def get_agent_raw(agent_id: str) -> dict | None:
    """Return the card as a plain dict (for diffing), or None if not found."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT card_json FROM agents WHERE id = %s", (agent_id,))
            row = cur.fetchone()
    else:
        with _sqlite_conn() as conn:
            row = conn.execute(
                "SELECT card_json FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def get_agent_with_health(agent_id: str) -> dict | None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT card_json, status, last_checked, last_seen_online "
                "FROM agents WHERE id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": agent_id,
            "agent_card": json.loads(row[0]),
            "health": {
                "status": row[1],
                "last_checked": str(row[2]) if row[2] else None,
                "last_seen_online": str(row[3]) if row[3] else None,
            },
        }
    else:
        with _sqlite_conn() as conn:
            row = conn.execute(
                "SELECT card_json, status, last_checked, last_seen_online "
                "FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": agent_id,
            "agent_card": json.loads(row["card_json"]),
            "health": {
                "status": row["status"],
                "last_checked": row["last_checked"],
                "last_seen_online": row["last_seen_online"],
            },
        }


def delete_agent(agent_id: str) -> bool:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM agents WHERE id = %s", (agent_id,))
            return cur.rowcount > 0
    else:
        with _sqlite_conn() as conn:
            cur = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        return cur.rowcount > 0


def list_all_agents() -> list[dict]:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, card_json, registered FROM agents ORDER BY registered DESC"
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "registered": str(r[2]), "agent_card": json.loads(r[1])}
            for r in rows
        ]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT id, card_json, registered FROM agents ORDER BY registered DESC"
            ).fetchall()
        return [
            {"id": r["id"], "registered": r["registered"], "agent_card": json.loads(r["card_json"])}
            for r in rows
        ]


def list_all_agents_with_health() -> list[dict]:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, card_json, registered, status, last_checked, last_seen_online "
                "FROM agents ORDER BY registered DESC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "registered": str(r[2]),
                "agent_card": json.loads(r[1]),
                "health": {
                    "status": r[3],
                    "last_checked": str(r[4]) if r[4] else None,
                    "last_seen_online": str(r[5]) if r[5] else None,
                },
            }
            for r in rows
        ]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT id, card_json, registered, status, last_checked, last_seen_online "
                "FROM agents ORDER BY registered DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "registered": r["registered"],
                "agent_card": json.loads(r["card_json"]),
                "health": {
                    "status": r["status"],
                    "last_checked": r["last_checked"],
                    "last_seen_online": r["last_seen_online"],
                },
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

def get_all_agent_urls_and_ids() -> list[dict]:
    """Return [{id, url, status, last_seen_online}] for the health monitor."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, card_json, status, last_seen_online FROM agents")
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "url": json.loads(r[1]).get("url", ""),
                "status": r[2],
                "last_seen_online": str(r[3]) if r[3] else None,
            }
            for r in rows
        ]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT id, card_json, status, last_seen_online FROM agents"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "url": json.loads(r["card_json"]).get("url", ""),
                "status": r["status"],
                "last_seen_online": r["last_seen_online"],
            }
            for r in rows
        ]


def update_agent_health(
    agent_id: str,
    status: str,
    last_checked: str,
    last_seen_online: str | None,
) -> None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE agents SET status=%s, last_checked=%s, last_seen_online=%s WHERE id=%s",
                (status, last_checked, last_seen_online, agent_id),
            )
    else:
        with _sqlite_conn() as conn:
            conn.execute(
                "UPDATE agents SET status=?, last_checked=?, last_seen_online=? WHERE id=?",
                (status, last_checked, last_seen_online, agent_id),
            )


def delete_stale_agents(cutoff_iso: str) -> list[str]:
    """Delete agents that have been offline/stale since before cutoff_iso."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM agents
                WHERE status IN ('offline', 'stale')
                  AND last_seen_online IS NOT NULL
                  AND last_seen_online::TEXT < %s
                RETURNING id
                """,
                (cutoff_iso,),
            )
            return [r[0] for r in cur.fetchall()]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                """
                SELECT id FROM agents
                WHERE status IN ('offline', 'stale')
                  AND last_seen_online IS NOT NULL
                  AND last_seen_online < ?
                """,
                (cutoff_iso,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM agents WHERE id IN ({placeholders})", ids)
            return ids


# ---------------------------------------------------------------------------
# Card versioning
# ---------------------------------------------------------------------------

def diff_cards(old: dict, new: dict) -> dict:
    """
    Field-by-field comparison. Returns {field: {old, new}} for changed fields.
    Uses JSON serialisation for consistent comparison of nested objects.
    """
    changed: dict = {}
    for key in set(old) | set(new):
        old_val = old.get(key)
        new_val = new.get(key)
        if json.dumps(old_val, sort_keys=True) != json.dumps(new_val, sort_keys=True):
            changed[key] = {"old": old_val, "new": new_val}
    return changed


def save_agent_version(agent_id: str, new_card_json: str, diff: dict) -> None:
    diff_json = json.dumps(diff)
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 FROM agent_versions WHERE agent_id = %s",
                (agent_id,),
            )
            version_num = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO agent_versions (agent_id, version_num, card_json, diff_json) "
                "VALUES (%s, %s, %s, %s)",
                (agent_id, version_num, new_card_json, diff_json),
            )
    else:
        with _sqlite_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 FROM agent_versions WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            version_num = row[0]
            conn.execute(
                "INSERT INTO agent_versions (agent_id, version_num, card_json, diff_json) "
                "VALUES (?, ?, ?, ?)",
                (agent_id, version_num, new_card_json, diff_json),
            )


def get_agent_versions(agent_id: str) -> list[dict]:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT version_num, card_json, diff_json, created_at "
                "FROM agent_versions WHERE agent_id = %s ORDER BY version_num ASC",
                (agent_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "version_num": r[0],
                "card": json.loads(r[1]),
                "diff": json.loads(r[2]),
                "created_at": str(r[3]),
            }
            for r in rows
        ]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT version_num, card_json, diff_json, created_at "
                "FROM agent_versions WHERE agent_id = ? ORDER BY version_num ASC",
                (agent_id,),
            ).fetchall()
        return [
            {
                "version_num": r["version_num"],
                "card": json.loads(r["card_json"]),
                "diff": json.loads(r["diff_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Search analytics
# ---------------------------------------------------------------------------

def log_search(query: str, results_count: int, tags: list[str]) -> None:
    tags_json = json.dumps(tags)
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO search_logs (query, results_count, tags_json) VALUES (%s, %s, %s)",
                (query, results_count, tags_json),
            )
    else:
        with _sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO search_logs (query, results_count, tags_json) VALUES (?, ?, ?)",
                (query, results_count, tags_json),
            )


def log_agent_click(agent_id: str, query_text: str | None = None) -> None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO agent_clicks (agent_id, query_text) VALUES (%s, %s)",
                (agent_id, query_text),
            )
    else:
        with _sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO agent_clicks (agent_id, query_text) VALUES (?, ?)",
                (agent_id, query_text),
            )


def get_analytics() -> dict:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT query, COUNT(*) AS cnt FROM search_logs "
                "GROUP BY query ORDER BY cnt DESC LIMIT 10"
            )
            top_queries = [{"query": r[0], "count": r[1]} for r in cur.fetchall()]
            cur.execute(
                "SELECT query, COUNT(*) AS cnt FROM search_logs WHERE results_count = 0 "
                "GROUP BY query ORDER BY cnt DESC LIMIT 10"
            )
            zero_result = [{"query": r[0], "count": r[1]} for r in cur.fetchall()]
            cur.execute(
                "SELECT agent_id, COUNT(*) AS cnt FROM agent_clicks "
                "GROUP BY agent_id ORDER BY cnt DESC LIMIT 10"
            )
            top_clicked = [{"agent_id": r[0], "clicks": r[1]} for r in cur.fetchall()]
    else:
        with _sqlite_conn() as conn:
            top_queries = [
                {"query": r["query"], "count": r["cnt"]}
                for r in conn.execute(
                    "SELECT query, COUNT(*) AS cnt FROM search_logs "
                    "GROUP BY query ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
            ]
            zero_result = [
                {"query": r["query"], "count": r["cnt"]}
                for r in conn.execute(
                    "SELECT query, COUNT(*) AS cnt FROM search_logs "
                    "WHERE results_count = 0 GROUP BY query ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
            ]
            top_clicked = [
                {"agent_id": r["agent_id"], "clicks": r["cnt"]}
                for r in conn.execute(
                    "SELECT agent_id, COUNT(*) AS cnt FROM agent_clicks "
                    "GROUP BY agent_id ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
            ]

    return {
        "top_queries": top_queries,
        "zero_result_queries": zero_result,
        "top_clicked_agents": top_clicked,
    }


# ---------------------------------------------------------------------------
# Ranking signals (for multi-signal reranking)
# ---------------------------------------------------------------------------

def log_impressions(agent_ids: list[str], query: str) -> None:
    """Record that each agent_id was shown as a search result for this query."""
    if not agent_ids:
        return
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT INTO agent_impressions (agent_id, query) VALUES (%s, %s)",
                [(aid, query) for aid in agent_ids],
            )
    else:
        with _sqlite_conn() as conn:
            conn.executemany(
                "INSERT INTO agent_impressions (agent_id, query) VALUES (?, ?)",
                [(aid, query) for aid in agent_ids],
            )


def update_task_centroid(agent_id: str, query_embedding: np.ndarray) -> None:
    """
    Online running-average update of the per-agent task centroid.
    new_centroid = (old_centroid * n + query_embedding) / (n + 1)
    """
    q = query_embedding.astype(np.float32)
    if _USE_POSTGRES:
        import psycopg2
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT centroid, click_count FROM agent_task_centroids WHERE agent_id = %s",
                (agent_id,),
            )
            row = cur.fetchone()
            if row is None:
                new_centroid = q
                new_count = 1
            else:
                old = np.frombuffer(bytes(row[0]), dtype=np.float32)
                n = row[1]
                new_centroid = (old * n + q) / (n + 1)
                new_count = n + 1
            blob = new_centroid.tobytes()
            cur.execute(
                """
                INSERT INTO agent_task_centroids (agent_id, centroid, click_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                    centroid    = EXCLUDED.centroid,
                    click_count = EXCLUDED.click_count
                """,
                (agent_id, psycopg2.Binary(blob), new_count),
            )
    else:
        with _sqlite_conn() as conn:
            row = conn.execute(
                "SELECT centroid, click_count FROM agent_task_centroids WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                new_centroid = q
                new_count = 1
            else:
                old = np.frombuffer(row["centroid"], dtype=np.float32)
                n = row["click_count"]
                new_centroid = (old * n + q) / (n + 1)
                new_count = n + 1
            blob = new_centroid.tobytes()
            conn.execute(
                """
                INSERT INTO agent_task_centroids (agent_id, centroid, click_count)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    centroid    = excluded.centroid,
                    click_count = excluded.click_count
                """,
                (agent_id, blob, new_count),
            )


def get_ranking_signals(agent_ids: list[str]) -> dict:
    """
    Bulk-fetch all ranking signals for the given agent_ids.

    Returns {agent_id: {ctr, recency_raw, task_centroid, avg_rating, registered_at, click_count}}
    where:
      - ctr           = clicks / max(impressions, 1)  [0, 1]
      - recency_raw   = raw decayed click sum (un-normalised)
      - task_centroid = np.ndarray | None
      - avg_rating    = float | None  (1–5)
      - registered_at = ISO string | None
      - click_count   = int
    """
    from ranking import compute_recency_raw

    if not agent_ids:
        return {}

    result: dict = {aid: {
        "ctr": 0.0, "recency_raw": 0.0, "task_centroid": None,
        "avg_rating": None, "registered_at": None, "click_count": 0,
    } for aid in agent_ids}

    if _USE_POSTGRES:
        ph = ",".join(["%s"] * len(agent_ids))
        with _pg_conn() as conn:
            cur = conn.cursor()

            # click timestamps per agent
            cur.execute(
                f"SELECT agent_id, clicked_at FROM agent_clicks WHERE agent_id IN ({ph})",
                agent_ids,
            )
            clicks_map: dict[str, list[str]] = {}
            for aid, ts in cur.fetchall():
                clicks_map.setdefault(aid, []).append(str(ts))

            # impression counts per agent
            cur.execute(
                f"SELECT agent_id, COUNT(*) FROM agent_impressions WHERE agent_id IN ({ph}) GROUP BY agent_id",
                agent_ids,
            )
            impressions_map = {r[0]: r[1] for r in cur.fetchall()}

            # task centroids
            cur.execute(
                f"SELECT agent_id, centroid FROM agent_task_centroids WHERE agent_id IN ({ph})",
                agent_ids,
            )
            centroids_map = {r[0]: np.frombuffer(bytes(r[1]), dtype=np.float32) for r in cur.fetchall()}

            # avg ratings
            cur.execute(
                f"SELECT agent_id, AVG(score) FROM agent_reviews WHERE agent_id IN ({ph}) GROUP BY agent_id",
                agent_ids,
            )
            ratings_map = {r[0]: float(r[1]) for r in cur.fetchall()}

            # registered_at
            cur.execute(
                f"SELECT id, registered FROM agents WHERE id IN ({ph})",
                agent_ids,
            )
            registered_map = {r[0]: str(r[1]) for r in cur.fetchall()}

    else:
        ph = ",".join(["?"] * len(agent_ids))
        with _sqlite_conn() as conn:
            clicks_map = {}
            for row in conn.execute(
                f"SELECT agent_id, clicked_at FROM agent_clicks WHERE agent_id IN ({ph})",
                agent_ids,
            ).fetchall():
                clicks_map.setdefault(row["agent_id"], []).append(row["clicked_at"])

            impressions_map = {
                r["agent_id"]: r["cnt"]
                for r in conn.execute(
                    f"SELECT agent_id, COUNT(*) AS cnt FROM agent_impressions WHERE agent_id IN ({ph}) GROUP BY agent_id",
                    agent_ids,
                ).fetchall()
            }
            centroids_map = {
                r["agent_id"]: np.frombuffer(r["centroid"], dtype=np.float32)
                for r in conn.execute(
                    f"SELECT agent_id, centroid FROM agent_task_centroids WHERE agent_id IN ({ph})",
                    agent_ids,
                ).fetchall()
            }
            ratings_map = {
                r["agent_id"]: float(r["avg_score"])
                for r in conn.execute(
                    f"SELECT agent_id, AVG(score) AS avg_score FROM agent_reviews WHERE agent_id IN ({ph}) GROUP BY agent_id",
                    agent_ids,
                ).fetchall()
            }
            registered_map = {
                r["id"]: r["registered"]
                for r in conn.execute(
                    f"SELECT id, registered FROM agents WHERE id IN ({ph})",
                    agent_ids,
                ).fetchall()
            }

    for aid in agent_ids:
        timestamps = clicks_map.get(aid, [])
        click_count = len(timestamps)
        impressions = impressions_map.get(aid, 0)
        result[aid] = {
            "ctr": click_count / max(impressions, 1),
            "recency_raw": compute_recency_raw(timestamps),
            "task_centroid": centroids_map.get(aid),
            "avg_rating": ratings_map.get(aid),
            "registered_at": registered_map.get(aid),
            "click_count": click_count,
        }

    return result


def submit_review(agent_id: str, reviewer_id: str, score: int, comment: str) -> dict:
    """Upsert a review (one review per reviewer per agent)."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO agent_reviews (agent_id, reviewer_id, score, comment)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (agent_id, reviewer_id) DO UPDATE SET
                    score   = EXCLUDED.score,
                    comment = EXCLUDED.comment,
                    created_at = NOW()
                RETURNING id, created_at
                """,
                (agent_id, reviewer_id, score, comment),
            )
            row = cur.fetchone()
            return {"id": row[0], "created_at": str(row[1])}
    else:
        with _sqlite_conn() as conn:
            conn.execute(
                """
                INSERT INTO agent_reviews (agent_id, reviewer_id, score, comment)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_id, reviewer_id) DO UPDATE SET
                    score      = excluded.score,
                    comment    = excluded.comment,
                    created_at = datetime('now')
                """,
                (agent_id, reviewer_id, score, comment),
            )
            row = conn.execute(
                "SELECT id, created_at FROM agent_reviews WHERE agent_id=? AND reviewer_id=?",
                (agent_id, reviewer_id),
            ).fetchone()
            return {"id": row["id"], "created_at": row["created_at"]}


def get_agent_reviews_list(agent_id: str) -> list[dict]:
    """Return all reviews for an agent, newest first."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT reviewer_id, score, comment, created_at "
                "FROM agent_reviews WHERE agent_id = %s ORDER BY created_at DESC",
                (agent_id,),
            )
            rows = cur.fetchall()
        return [
            {"reviewer_id": r[0], "score": r[1], "comment": r[2], "created_at": str(r[3])}
            for r in rows
        ]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT reviewer_id, score, comment, created_at "
                "FROM agent_reviews WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        return [
            {"reviewer_id": r["reviewer_id"], "score": r["score"],
             "comment": r["comment"], "created_at": r["created_at"]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Vector index (in-RAM cosine similarity)
# ---------------------------------------------------------------------------

def load_vector_index() -> tuple[list[str], np.ndarray | None]:
    """Return (ids, matrix) where matrix rows correspond to ids."""
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, embedding FROM agents")
            rows = cur.fetchall()
        if not rows:
            return [], None
        ids = [r[0] for r in rows]
        vectors = [np.frombuffer(bytes(r[1]), dtype=np.float32) for r in rows]
    else:
        with _sqlite_conn() as conn:
            rows = conn.execute("SELECT id, embedding FROM agents").fetchall()
        if not rows:
            return [], None
        ids = [r["id"] for r in rows]
        vectors = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    return ids, np.vstack(vectors)


def cosine_search(
    query_vec: np.ndarray,
    top_k: int = 5,
    tag_filter: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Return [(agent_id, score)] sorted by descending cosine similarity."""
    ids, matrix = load_vector_index()
    if not ids or matrix is None:
        return []

    q = query_vec.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    scores: np.ndarray = (matrix / norms) @ q

    if tag_filter:
        tag_set = {t.lower() for t in tag_filter}
        all_agents = list_all_agents()
        id_to_tags: dict[str, set] = {}
        for a in all_agents:
            card = a["agent_card"]
            agent_tags = {t.lower() for t in card.get("tags", [])}
            for skill in card.get("skills", []):
                agent_tags.update(t.lower() for t in skill.get("tags", []))
            id_to_tags[a["id"]] = agent_tags

        for i, aid in enumerate(ids):
            if not (tag_set & id_to_tags.get(aid, set())):
                scores[i] = -1.0

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def make_agent_id(card: AgentCard) -> str:
    if card.humanReadableId:
        return card.humanReadableId.replace("/", "__")
    return str(uuid.uuid4())
