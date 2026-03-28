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
    clicked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_clicks_agent_id ON agent_clicks (agent_id);
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
    clicked_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_clicks_agent_id ON agent_clicks (agent_id);
"""


def _migrate_agents_columns(conn, use_postgres: bool) -> None:
    """Add health columns to agents tables that predate this schema version."""
    new_cols = [
        ("status",           "TEXT NOT NULL DEFAULT 'unknown'"),
        ("last_checked",     "TIMESTAMPTZ" if use_postgres else "TEXT"),
        ("last_seen_online", "TIMESTAMPTZ" if use_postgres else "TEXT"),
    ]
    if use_postgres:
        cur = conn.cursor()
        for col, typedef in new_cols:
            cur.execute(f"ALTER TABLE agents ADD COLUMN IF NOT EXISTS {col} {typedef}")
    else:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(agents)").fetchall()}
        for col, typedef in new_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE agents ADD COLUMN {col} {typedef}")


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


def log_agent_click(agent_id: str) -> None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO agent_clicks (agent_id) VALUES (%s)", (agent_id,))
    else:
        with _sqlite_conn() as conn:
            conn.execute("INSERT INTO agent_clicks (agent_id) VALUES (?)", (agent_id,))


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
