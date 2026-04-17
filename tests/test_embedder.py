"""Tests for src.embedder."""

from unittest.mock import patch, MagicMock

from src.embedder import embed_documents, embed_query

_FAKE_VECTOR = [0.1, 0.2, 0.3]


def test_embed_documents() -> None:
    """embed_documents returns a list of float vectors."""
    with patch.object(
        __import__("src.embedder", fromlist=["_cached"]).
        _cached, "embed_documents", return_value=[_FAKE_VECTOR, _FAKE_VECTOR]
    ):
        result = embed_documents(["hello", "world"])

    assert len(result) == 2
    assert all(isinstance(v, list) and all(isinstance(f, float) for f in v) for v in result)


def test_embed_query() -> None:
    """embed_query returns a single float vector."""
    with patch.object(
        __import__("src.embedder", fromlist=["_cached"]).
        _cached, "embed_query", return_value=_FAKE_VECTOR
    ):
        result = embed_query("hello")

    assert isinstance(result, list)
    assert all(isinstance(f, float) for f in result)
