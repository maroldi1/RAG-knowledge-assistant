"""Tests for src.rag_chain."""

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import src.rag_chain as rc


def test_invoke_returns_answer_and_sources():
    """invoke() returns dict with 'answer' and 'sources' keys."""
    mock_docs = [
        Document(page_content="Policy text", metadata={"source": "policy.pdf"}),
    ]

    with (
        patch.object(rc, "embed_query", return_value=[0.1, 0.2, 0.3]),
        patch.object(rc, "similarity_search", return_value=mock_docs),
        patch("langchain_openai.AzureChatOpenAI.invoke", return_value=AIMessage(content="The answer is 42.")),
    ):
        result = rc.invoke("What is the answer?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "The answer is 42."
    assert "policy.pdf" in result["sources"]


def test_system_prompt_contains_context():
    """The system prompt instructs the model to use context."""
    assert "context" in rc._SYSTEM_PROMPT.lower()
