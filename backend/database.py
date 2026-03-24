"""
SQLite-backed store for agent cards + numpy vector index for semantic search.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from models import AgentCard

DB_PATH = Path(__file__).parent / "agents.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id          TEXT PRIMARY KEY,
    card_json   TEXT NOT NULL,
    embedding   BLOB NOT NULL,          -- serialised float32 numpy array
    registered  TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert_agent(agent_id: str, card: AgentCard, embedding: np.ndarray) -> str:
    blob = embedding.astype(np.float32).tobytes()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO agents (id, card_json, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                card_json  = excluded.card_json,
                embedding  = excluded.embedding,
                registered = datetime('now')
            """,
            (agent_id, card.model_dump_json(), blob),
        )
    return agent_id


def get_agent(agent_id: str) -> AgentCard | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT card_json FROM agents WHERE id = ?", (agent_id,)
        ).fetchone()
    if row is None:
        return None
    return AgentCard.model_validate_json(row["card_json"])


def delete_agent(agent_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
    return cur.rowcount > 0


def list_all_agents() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, card_json, registered FROM agents ORDER BY registered DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "registered": r["registered"],
            "agent_card": json.loads(r["card_json"]),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Vector index (all-in-RAM cosine similarity; fine for thousands of agents)
# ---------------------------------------------------------------------------

def load_vector_index() -> tuple[list[str], np.ndarray | None]:
    """Return (ids, matrix) where matrix rows correspond to ids."""
    with _conn() as conn:
        rows = conn.execute("SELECT id, embedding FROM agents").fetchall()
    if not rows:
        return [], None
    ids = [r["id"] for r in rows]
    vectors = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    matrix = np.vstack(vectors)  # (N, D)
    return ids, matrix


def cosine_search(
    query_vec: np.ndarray,
    top_k: int = 5,
    tag_filter: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Return [(agent_id, score)] sorted by descending cosine similarity."""
    ids, matrix = load_vector_index()
    if not ids or matrix is None:
        return []

    # Normalise
    q = query_vec.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed = matrix / norms
    scores: np.ndarray = normed @ q  # (N,)

    # Optional tag pre-filter — zero-out non-matching rows
    if tag_filter:
        tag_set = {t.lower() for t in tag_filter}
        with _conn() as conn:
            rows = conn.execute("SELECT id, card_json FROM agents").fetchall()
        id_to_tags: dict[str, set] = {}
        for r in rows:
            card = json.loads(r["card_json"])
            agent_tags = {t.lower() for t in card.get("tags", [])}
            for skill in card.get("skills", []):
                agent_tags.update(t.lower() for t in skill.get("tags", []))
            id_to_tags[r["id"]] = agent_tags

        for i, aid in enumerate(ids):
            if not (tag_set & id_to_tags.get(aid, set())):
                scores[i] = -1.0

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def make_agent_id(card: AgentCard) -> str:
    """Stable ID derived from humanReadableId, or a fresh UUID."""
    if card.humanReadableId:
        safe = card.humanReadableId.replace("/", "__")
        return safe
    return str(uuid.uuid4())
