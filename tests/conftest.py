"""Set test environment variables and patch heavy imports before any src module loads."""

import os
from unittest.mock import MagicMock, patch

# Environment variables needed by src.config at import time
os.environ.setdefault("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("E2B_API_KEY", "e2b_test")
os.environ.setdefault("DATABASE_URL", "postgresql://u:p@localhost:5433/test")

# Patch HuggingFaceEmbeddings before any src module triggers a real model download
_mock_hf_instance = MagicMock()
_mock_hf_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
_mock_hf_instance.embed_query.return_value = [0.1, 0.2, 0.3]
_hf_patcher = patch(
    "langchain_huggingface.HuggingFaceEmbeddings",
    return_value=_mock_hf_instance,
)
_hf_patcher.start()

# Patch HuggingFacePipeline and ChatHuggingFace to prevent real model loading
_mock_pipeline = MagicMock()
_mock_pipeline.invoke.return_value = "mocked response"
_pipeline_patcher = patch(
    "langchain_huggingface.HuggingFacePipeline.from_model_id",
    return_value=_mock_pipeline,
)
_pipeline_patcher.start()

_mock_chat = MagicMock()
_chat_patcher = patch(
    "langchain_huggingface.ChatHuggingFace",
    return_value=_mock_chat,
)
_chat_patcher.start()
