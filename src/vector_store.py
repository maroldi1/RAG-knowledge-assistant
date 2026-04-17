"""pgvector store helpers."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from src import config
from src.embedder import _cached as _embeddings

logger = logging.getLogger(__name__)

_COLLECTION = "knowledge_base"
_store: Optional[PGVector] = None


def get_store() -> PGVector:
    """Return a singleton PGVector instance, creating tables if needed."""
    global _store
    if _store is not None:
        return _store

    _store = PGVector(
        embeddings=_embeddings,
        collection_name=_COLLECTION,
        connection=config.DATABASE_URL,
        distance_strategy=DistanceStrategy.COSINE,
    )
    _store.create_vector_extension()
    _store.create_tables_if_not_exists()
    return _store


def add_documents(docs: list[Document]) -> None:
    """Upsert document chunks, skipping duplicates by source + chunk_index."""
    store = get_store()
    new_docs: list[Document] = []
    ids: list[str] = []

    for doc in docs:
        doc_id = f"{doc.metadata['source']}::{doc.metadata['chunk_index']}"
        ids.append(doc_id)
        new_docs.append(doc)

    store.add_documents(new_docs, ids=ids)
    logger.info("Upserted %d documents", len(new_docs))


def similarity_search(query_embedding: list[float], k: int) -> list[Document]:
    """Return top-k documents by cosine similarity to the query embedding."""
    store = get_store()
    return store.similarity_search_by_vector(query_embedding, k=k)
