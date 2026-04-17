"""Tests for src.vector_store."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import src.vector_store as vs


@patch.object(vs, "PGVector")
def test_add_documents(mock_pg_cls):
    """add_documents passes docs and deterministic IDs to the store."""
    mock_store = MagicMock()
    mock_pg_cls.return_value = mock_store
    vs._store = None

    docs = [
        Document(page_content="chunk 0", metadata={"source": "a.txt", "chunk_index": 0}),
        Document(page_content="chunk 1", metadata={"source": "a.txt", "chunk_index": 1}),
    ]

    vs.add_documents(docs)

    mock_store.add_documents.assert_called_once()
    call_args = mock_store.add_documents.call_args
    assert len(call_args[0][0]) == 2
    assert call_args[1]["ids"] == ["a.txt::0", "a.txt::1"]

    vs._store = None


@patch.object(vs, "PGVector")
def test_similarity_search(mock_pg_cls):
    """similarity_search delegates to the store's similarity_search_by_vector."""
    mock_store = MagicMock()
    expected = [Document(page_content="result", metadata={"source": "b.txt"})]
    mock_store.similarity_search_by_vector.return_value = expected
    mock_pg_cls.return_value = mock_store
    vs._store = None

    embedding = [0.1, 0.2, 0.3]
    results = vs.similarity_search(embedding, k=3)

    mock_store.similarity_search_by_vector.assert_called_once_with(embedding, k=3)
    assert results == expected

    vs._store = None
