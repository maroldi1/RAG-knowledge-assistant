"""Embedding helpers using HuggingFace via LangChain with in-memory caching."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryByteStore
from langchain.embeddings import CacheBackedEmbeddings

from src import config

logger = logging.getLogger(__name__)
logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)

_underlying = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
_store = InMemoryByteStore()
_cached = CacheBackedEmbeddings.from_bytes_store(
    _underlying,
    _store,
    namespace=config.EMBEDDING_MODEL,
)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, using the cache when possible."""
    return _cached.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return _cached.embed_query(text)
