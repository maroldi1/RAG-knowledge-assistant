"""Set test environment variables before any src module is imported."""

import os

os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-azure-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
os.environ.setdefault("AZURE_OPENAI_CHAT_DEPLOYMENT", "test-chat")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "test-embedding")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("E2B_API_KEY", "e2b_test")
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5433/test")
