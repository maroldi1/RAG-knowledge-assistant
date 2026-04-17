"""RAG chain — retrieval, prompt assembly, and HuggingFace LLM invocation via LCEL."""

import logging

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src import config
from src.embedder import embed_query
from src.vector_store import similarity_search

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an internal knowledge assistant. Answer the user's question using "
    "ONLY the provided context below. For every claim you make, cite the source "
    "filename in square brackets, e.g. [refund_policy.pdf]. If the context does "
    "not contain enough information to answer, say so explicitly — do not guess."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

try:
    _pipeline = HuggingFacePipeline.from_model_id(
        model_id=config.HF_LLM_MODEL,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "temperature": 0.2,
            "do_sample": True,
        },
    )
    _llm = ChatHuggingFace(llm=_pipeline)
except (OSError, ValueError) as exc:
    raise RuntimeError(
        f"Failed to load local model '{config.HF_LLM_MODEL}': {exc}"
    ) from exc


def _retrieve(question: str) -> list[Document]:
    """Embed the question and retrieve top-k similar chunks."""
    embedding = embed_query(question)
    return similarity_search(embedding, k=config.TOP_K_RESULTS)


def _format_context(docs: list[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_inputs(question: str) -> dict:
    """Retrieve docs and build the prompt inputs dict."""
    docs = _retrieve(question)
    return {
        "context": _format_context(docs),
        "question": question,
        "_docs": docs,
    }


_chain = (
    RunnableLambda(_build_inputs)
    | RunnablePassthrough.assign(
        answer=_prompt.partial() | _llm | StrOutputParser()
    )
)


def invoke(question: str) -> dict[str, object]:
    """Run the RAG chain and return answer with source filenames."""
    result = _chain.invoke(question)
    sources = list({d.metadata.get("source", "unknown") for d in result["_docs"]})
    return {"answer": result["answer"], "sources": sorted(sources)}
