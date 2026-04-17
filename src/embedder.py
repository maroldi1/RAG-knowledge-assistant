"""Embedding helpers using Azure OpenAI via LangChain with in-memory caching."""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.stores import InMemoryByteStore
from langchain.embeddings import CacheBackedEmbeddings

from src import config

_underlying = AzureOpenAIEmbeddings(
    azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
    api_version=config.AZURE_OPENAI_API_VERSION,
    model=config.EMBEDDING_MODEL,
)
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
