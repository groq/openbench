import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from openbench.utils.factscore_wiki import (
    KnowledgeSourceError,
    WikipediaRetriever,
    SPECIAL_SEPARATOR,
)


def _create_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "wiki.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE documents (title TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        paragraphs = ["Paragraph one about Topic.", "Second paragraph with details."]
        connection.execute(
            "INSERT INTO documents (title, text) VALUES (?, ?)",
            ("Test Topic", SPECIAL_SEPARATOR.join(paragraphs)),
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


def test_wikipedia_retriever_selects_passage(tmp_path):
    db_path = _create_db(tmp_path)
    cache_path = tmp_path / "cache.json"

    retriever = WikipediaRetriever(
        db_path=db_path, cache_path=cache_path, num_passages=1
    )

    knowledge = retriever.get_knowledge("Test Topic", "bio")
    assert "Paragraph" in knowledge
    # Retrieval should cache results for subsequent calls
    cached = retriever.get_knowledge("Test Topic", "bio")
    assert knowledge == cached


def test_wikipedia_retriever_topic_variants(tmp_path):
    db_path = _create_db(tmp_path)
    retriever = WikipediaRetriever(
        db_path=db_path, cache_path=tmp_path / "cache.json", num_passages=2
    )

    knowledge = retriever.get_knowledge("Test_Topic", None)
    assert "Paragraph" in knowledge


def test_wikipedia_retriever_missing_topic(tmp_path):
    db_path = _create_db(tmp_path)
    retriever = WikipediaRetriever(
        db_path=db_path, cache_path=tmp_path / "cache.json", num_passages=2
    )

    with pytest.raises(KnowledgeSourceError):
        retriever.get_knowledge("Unknown", None)
