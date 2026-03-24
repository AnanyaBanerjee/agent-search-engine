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
    id          TEXT PRIMARY KEY,
    card_json   TEXT        NOT NULL,
    embedding   BYTEA       NOT NULL,
    registered  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    card_json   TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    registered  TEXT DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    if _USE_POSTGRES:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(_SCHEMA_PG)
    else:
        with _sqlite_conn() as conn:
            conn.executescript(_SCHEMA_SQLITE)


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
