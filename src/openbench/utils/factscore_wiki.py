"""Lightweight Wikipedia retrieval utilities for FactScoreLite.

These helpers load the official FActScore SQLite dump, extract candidate
paragraphs for a given topic, and optionally rank them before returning a
condensed knowledge string suitable for FactScoreLite's scorer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class KnowledgeSourceError(RuntimeError):
    """Raised when a requested Wikipedia article cannot be located."""


def _topic_variants(topic: str) -> Iterable[str]:
    """Return normalised variants to improve matching against SQLite titles."""

    stripped = topic.strip()
    yield stripped
    if "_" in stripped:
        yield stripped.replace("_", " ")
    if " " in stripped:
        yield stripped.replace(" ", "_")


class _WikipediaConnection:
    """Thread-safe wrapper around sqlite3 connection access."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()

    def fetch_paragraphs(self, topic: str) -> list[str]:
        """Return decoded paragraphs for ``topic`` or raise if not found."""

        with self._lock:
            cursor = self._connection.cursor()
            try:
                for candidate in _topic_variants(topic):
                    cursor.execute(
                        "SELECT text FROM documents WHERE title = ?", (candidate,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        continue
                    paragraphs = [
                        paragraph.strip()
                        for paragraph in row[0].split(SPECIAL_SEPARATOR)
                        if paragraph and paragraph.strip()
                    ]
                    if paragraphs:
                        return paragraphs
            finally:
                cursor.close()

        raise KnowledgeSourceError(f"No Wikipedia entry found for topic '{topic}'.")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _select_passages(
    topic: str,
    query: str | None,
    passages: Sequence[str],
    num_passages: int | None,
) -> list[str]:
    """Select up to ``num_passages`` passages using BM25 scoring."""

    if num_passages is None or len(passages) <= num_passages:
        return list(passages)

    tokenised_passages: list[list[str]] = []
    for passage in passages:
        tokens = passage.replace("<s>", " ").replace("</s>", " ").split()
        tokenised_passages.append(tokens)

    if not any(tokenised_passages):
        return list(passages[:num_passages])

    try:
        from rank_bm25 import (  # type: ignore[import-not-found, import-untyped]
            BM25Okapi,
        )
    except Exception as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "BM25 ranking requires the 'rank-bm25' package. Install with `pip install openbench[factscore]`."
        ) from exc

    bm25 = BM25Okapi(tokenised_passages)
    query_tokens = []
    if query:
        query_tokens = query.split()
    if not query_tokens:
        query_tokens = topic.split()
    if not query_tokens:
        return list(passages[:num_passages])

    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores)[:num_passages]
    return [passages[i] for i in order]


@dataclass
class WikipediaRetriever:
    """Cached accessor that condenses Wikipedia articles into knowledge strings."""

    db_path: Path
    cache_path: Path
    num_passages: int = 8

    def __post_init__(self) -> None:
        self._conn = _WikipediaConnection(self.db_path)
        self._memory_cache: Dict[str, str] = {}
        self._cache_path = self.cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._memory_cache.update({str(k): str(v) for k, v in data.items()})
            except json.JSONDecodeError:
                # Corrupt cache, ignore and rebuild on the fly.
                pass

    def _save(self) -> None:
        if not self._dirty:
            return
        self._cache_path.write_text(json.dumps(self._memory_cache), encoding="utf-8")
        self._dirty = False

    def get_knowledge(self, topic: str, query: str | None = None) -> str:
        topic_key = topic.strip()
        if not topic_key:
            raise KnowledgeSourceError(
                "Cannot retrieve knowledge without a topic identifier."
            )

        cached = self._memory_cache.get(topic_key)
        if cached is not None:
            return cached

        paragraphs = self._conn.fetch_paragraphs(topic_key)
        selected = _select_passages(topic_key, query, paragraphs, self.num_passages)
        if not selected:
            raise KnowledgeSourceError(
                f"No passages extracted for topic '{topic_key}'."
            )

        knowledge = "\n\n".join(selected)
        self._memory_cache[topic_key] = knowledge
        self._dirty = True
        return knowledge

    def close(self) -> None:
        self._save()
        self._conn.close()


__all__ = [
    "WikipediaRetriever",
    "KnowledgeSourceError",
]
