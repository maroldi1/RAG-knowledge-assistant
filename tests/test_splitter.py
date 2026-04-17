"""Tests for src.splitter."""

import os
import tempfile

from langchain_core.documents import Document

from src.splitter import load_and_split


def test_load_and_split_txt():
    """load_and_split returns chunks with correct metadata keys from a .txt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.txt")
        with open(path, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 50)

        chunks = load_and_split(tmpdir)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "ingested_at" in chunk.metadata
        assert chunk.metadata["source"] == "sample.txt"
        assert isinstance(chunk.metadata["chunk_index"], int)
