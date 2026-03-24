"""
Embedding-based semantic search using fastembed (ONNX runtime, no PyTorch).
Model is loaded once at startup and reused.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # ~130 MB, fast, high quality


@lru_cache(maxsize=1)
def _get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    model = _get_model()
    vectors = list(model.embed([text]))
    return np.array(vectors[0], dtype=np.float32)


def build_agent_text(card_dict: dict) -> str:
    """
    Concatenate all searchable fields from an agent card into one string
    for embedding. More weight is given to description and skill descriptions.
    """
    parts: list[str] = []

    if name := card_dict.get("name"):
        parts.append(name)

    if desc := card_dict.get("description"):
        parts.append(desc)
        parts.append(desc)          # repeat for extra weight

    for tag in card_dict.get("tags", []):
        parts.append(tag)

    for skill in card_dict.get("skills", []):
        if sname := skill.get("name"):
            parts.append(sname)
        if sdesc := skill.get("description"):
            parts.append(sdesc)
            parts.append(sdesc)     # repeat for extra weight
        for stag in skill.get("tags", []):
            parts.append(stag)
        for ex in skill.get("examples", []):
            parts.append(ex)

    return " ".join(parts)


def embed_agent_card(card_dict: dict) -> np.ndarray:
    return embed_text(build_agent_text(card_dict))
