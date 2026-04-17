"""Tests for src.rag_chain."""

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import src.rag_chain as rc


def test_invoke_returns_answer_and_sources():
    """invoke() returns dict with 'answer' and 'sources' keys."""
    mock_docs = [
        Document(page_content="Policy text", metadata={"source": "policy.pdf"}),
    ]

    with (
        patch.object(rc, "_retrieve", return_value=mock_docs),
        patch.object(rc, "_format_context", return_value="[policy.pdf]\nPolicy text"),
    ):
        # Rebuild the chain with a fake LLM so the LCEL pipeline works end-to-end
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        fake_llm = RunnableLambda(lambda _: AIMessage(content="The answer is 42."))
        rc._chain = (
            RunnableLambda(rc._build_inputs)
            | RunnablePassthrough.assign(
                answer=rc._prompt.partial() | fake_llm | StrOutputParser()
            )
        )
        result = rc.invoke("What is the answer?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "The answer is 42."
    assert "policy.pdf" in result["sources"]


def test_system_prompt_contains_context():
    """The system prompt instructs the model to use context."""
    assert "context" in rc._SYSTEM_PROMPT.lower()
