"""Application configuration — reads and validates environment variables."""

import logging
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_REQUIRED = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "E2B_API_KEY",
    "DATABASE_URL",
]

for _key in _REQUIRED:
    if not os.getenv(_key):
        raise ValueError(f"Missing required environment variable: {_key}")

AZURE_OPENAI_API_KEY: str = os.environ["AZURE_OPENAI_API_KEY"]
_raw_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
_parsed = urlparse(_raw_endpoint)
if _parsed.path not in ("", "/") or _parsed.query:
    logger.warning(
        "AZURE_OPENAI_ENDPOINT should be a base URL (e.g. https://host.openai.azure.com/). "
        "Stripping path and query from: %s",
        _raw_endpoint,
    )
AZURE_OPENAI_ENDPOINT: str = f"{_parsed.scheme}://{_parsed.netloc}/"
AZURE_OPENAI_API_VERSION: str = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
E2B_API_KEY: str = os.environ["E2B_API_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
