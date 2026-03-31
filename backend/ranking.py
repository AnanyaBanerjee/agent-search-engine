"""
Agent ranking / re-ranking module.

Blends five signals into a final score for each candidate agent:

  final = w_semantic  * semantic_score      (cosine similarity)
        + w_ctr       * ctr_score           (clicks / impressions, log-normalised)
        + w_recency   * recency_score        (exponentially decayed recent clicks)
        + w_affinity  * task_affinity_score  (cosine sim: query vs click-query centroid)
        + w_reputation* reputation_score     (avg review rating)
        + cold_start_bonus                   (temporary boost for new/unseen agents)

All weights are configurable via environment variables.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Weight configuration (tunable via env vars)
# ---------------------------------------------------------------------------

W_SEMANTIC      = float(os.environ.get("RANK_WEIGHT_SEMANTIC",      0.50))
W_CTR           = float(os.environ.get("RANK_WEIGHT_CTR",           0.20))
W_RECENCY       = float(os.environ.get("RANK_WEIGHT_RECENCY",       0.15))
W_AFFINITY      = float(os.environ.get("RANK_WEIGHT_TASK_AFFINITY", 0.10))
W_REPUTATION    = float(os.environ.get("RANK_WEIGHT_REPUTATION",    0.05))

DECAY_LAMBDA            = float(os.environ.get("RANK_RECENCY_DECAY_LAMBDA",  0.10))
COLD_START_BONUS        = float(os.environ.get("RANK_COLD_START_BONUS",      0.10))
COLD_START_MIN_CLICKS   = int(os.environ.get("RANK_COLD_START_MIN_CLICKS",   5))
COLD_START_MAX_DAYS     = int(os.environ.get("RANK_COLD_START_MAX_DAYS",     7))

# How many extra candidates to fetch from cosine search before reranking
RERANK_POOL_MULTIPLIER  = int(os.environ.get("RANK_POOL_MULTIPLIER",         3))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime | None:
    try:
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def compute_recency_raw(click_timestamps: list[str]) -> float:
    """
    Exponentially decayed sum of clicks.
    Each click contributes exp(-λ × days_since_click).
    Older clicks contribute less; clicks today contribute ~1.0 each.
    """
    now = datetime.now(timezone.utc)
    total = 0.0
    for ts in click_timestamps:
        dt = _parse_iso(ts)
        if dt is None:
            continue
        days_ago = (now - dt).total_seconds() / 86400
        total += math.exp(-DECAY_LAMBDA * days_ago)
    return total


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a) + 1e-9
    norm_b = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_cold_start(registered_at: str | None, click_count: int) -> bool:
    if click_count >= COLD_START_MIN_CLICKS:
        return False
    if registered_at is None:
        return True
    dt = _parse_iso(registered_at)
    if dt is None:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=COLD_START_MAX_DAYS)
    return dt > cutoff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    hits: list[tuple[str, float]],
    query_embedding: Optional[np.ndarray],
    signals: dict,
) -> list[tuple[str, float]]:
    """
    Re-rank a list of (agent_id, semantic_score) pairs using all available signals.

    `signals` is a dict keyed by agent_id, with fields:
        ctr             float   clicks / impressions (0–1)
        recency_raw     float   decayed click sum (un-normalised)
        task_centroid   ndarray | None  per-agent click-query centroid
        avg_rating      float | None    mean review score (1–5)
        registered_at   str | None      ISO-8601 registration timestamp
        click_count     int

    Returns the same list re-sorted by final composite score, descending.
    """
    if not hits:
        return hits

    # --- normalise across the current candidate set ---
    max_recency = max(
        (signals.get(aid, {}).get("recency_raw", 0.0) for aid, _ in hits),
        default=1.0,
    ) or 1.0
    max_ctr = max(
        (signals.get(aid, {}).get("ctr", 0.0) for aid, _ in hits),
        default=1.0,
    ) or 1.0

    scored: list[tuple[str, float]] = []

    for agent_id, semantic_score in hits:
        s = signals.get(agent_id, {})

        # 1. CTR (normalised to [0, 1] within candidate set)
        ctr_norm = s.get("ctr", 0.0) / max_ctr

        # 2. Recency (normalised)
        recency_norm = s.get("recency_raw", 0.0) / max_recency

        # 3. Task affinity — cosine between current query and per-agent centroid
        affinity = 0.0
        centroid = s.get("task_centroid")
        if query_embedding is not None and centroid is not None:
            try:
                affinity = max(0.0, _cosine(query_embedding.astype(np.float32), centroid))
            except Exception:
                pass

        # 4. Reputation (avg rating 1–5 → 0–1)
        avg_rating = s.get("avg_rating")
        reputation = (avg_rating / 5.0) if avg_rating is not None else 0.0

        # 5. Cold-start bonus for new / unseen agents
        cold_bonus = (
            COLD_START_BONUS
            if _is_cold_start(s.get("registered_at"), s.get("click_count", 0))
            else 0.0
        )

        final = (
            W_SEMANTIC   * semantic_score
            + W_CTR      * ctr_norm
            + W_RECENCY  * recency_norm
            + W_AFFINITY * affinity
            + W_REPUTATION * reputation
            + cold_bonus
        )
        scored.append((agent_id, round(final, 6)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
