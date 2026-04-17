"""Application configuration — reads and validates environment variables."""

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_REQUIRED = [
    "HF_LLM_MODEL",
    "E2B_API_KEY",
    "DATABASE_URL",
]

for _key in _REQUIRED:
    if not os.getenv(_key):
        raise ValueError(f"Missing required environment variable: {_key}")

HF_LLM_MODEL: str = os.environ["HF_LLM_MODEL"]
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
E2B_API_KEY: str = os.environ["E2B_API_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
