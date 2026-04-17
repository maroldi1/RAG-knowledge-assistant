"""Document loading and text splitting."""

import logging
import os
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

_LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)


def load_and_split(docs_dir: str) -> list[Document]:
    """Load all supported files from docs_dir and split into chunks."""
    all_chunks: list[Document] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for filename in sorted(os.listdir(docs_dir)):
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = _LOADERS.get(ext)
        if loader_cls is None:
            logger.debug("Skipping unsupported file: %s", filename)
            continue

        filepath = os.path.join(docs_dir, filename)
        logger.info("Loading %s", filepath)
        docs = loader_cls(filepath).load()
        chunks = _splitter.split_documents(docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": filename,
                "page": chunk.metadata.get("page"),
                "chunk_index": idx,
                "ingested_at": ingested_at,
            }

        all_chunks.extend(chunks)

    logger.info("Loaded %d chunks from %s", len(all_chunks), docs_dir)
    return all_chunks
