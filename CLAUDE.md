# RAG Knowledge Assistant — CLAUDE.md

This file guides Claude Code when generating and editing code for this project.
Read it fully before writing any file.

---

## Project overview

An internal knowledge assistant built with:
- **LangChain** — orchestration (ingestion, retrieval, prompt assembly)
- **pgvector** — PostgreSQL extension for vector similarity search
- **E2B** — sandboxed Python execution environment
- **HuggingFace** — local embeddings (sentence-transformers) and local chat completion (transformers pipeline)
- **Python 3.11+**

The system has two runtime modes:
1. **Ingestion mode** — chunk documents, embed them, store vectors in pgvector
2. **Query mode** — embed the user question, retrieve top-k chunks, call HuggingFace LLM

---

## Project structure

```
rag-assistant/
├── CLAUDE.md                  ← this file
├── .env.example               ← environment variable template
├── requirements.txt           ← pinned dependencies
├── scripts/
│   ├── ingest.py              ← ingestion pipeline (run once per corpus)
│   ├── query.py               ← interactive query loop
│   └── sandbox_runner.py      ← wraps both scripts inside an E2B sandbox
├── src/
│   ├── __init__.py
│   ├── config.py              ← reads env vars, validates at startup
│   ├── embedder.py            ← LangChain embedding wrapper
│   ├── splitter.py            ← document loading + text splitting
│   ├── vector_store.py        ← pgvector connection + CRUD helpers
│   └── rag_chain.py           ← LangChain RAG chain + HuggingFace LLM integration
├── tests/
│   ├── test_splitter.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   └── test_rag_chain.py
└── docker-compose.yml         ← spins up PostgreSQL + pgvector locally
```

---

## Environment variables

All secrets live in `.env` (never committed). Copy `.env.example` to `.env`:

```
HF_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
E2B_API_KEY=e2b_...
DATABASE_URL=postgresql://user:password@localhost:5433/rag_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
MAX_NEW_TOKENS=512
```

`src/config.py` must raise a clear `ValueError` at import time if any required
variable is missing.

---

## Code conventions

### General
- Python 3.11+, type hints on every function signature
- `black` formatting, line length 88
- `ruff` for linting
- All functions have a one-line docstring
- No hardcoded secrets or paths — always use `config.py`

### LangChain
- Use `langchain-community` and `langchain-core` — avoid the deprecated
  `langchain` monolith package
- Prefer `LCEL` (LangChain Expression Language) chains using the `|` pipe operator
- Use `RecursiveCharacterTextSplitter` for chunking
- Wrap embeddings in a `CacheBackedEmbeddings` layer to avoid re-embedding
  unchanged documents

### pgvector
- Use `PGVector` from `langchain-postgres` (not the older `langchain-community` version)
- Connection string from `config.DATABASE_URL`
- Collection name: `knowledge_base`
- Distance strategy: `DistanceStrategy.COSINE`
- Always call `.create_tables_if_not_exist()` at startup

### HuggingFace
- Use `HuggingFaceEmbeddings` from `langchain-huggingface` for local embeddings
- Use `HuggingFacePipeline` from `langchain-huggingface` for local LLM inference,
  wrapped in `ChatHuggingFace` for chat-style prompts
- Embedding model: `config.EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- LLM model: `config.HF_LLM_MODEL` (required, e.g. `mistralai/Mistral-7B-Instruct-v0.3`)
- All models run locally — no API tokens required
- Temperature: `0.2` for factual Q&A
- Include a system prompt that instructs the model to cite which document
  chunk it used
- `max_new_tokens`: 512 (adjustable via env var `MAX_NEW_TOKENS`)

### E2B sandbox
- Use the `e2b` Python SDK
- Create one sandbox per session using `Sandbox(template="base")`
- Upload scripts and any local documents via `sandbox.files.write()`
- Stream stdout back to the caller using `sandbox.run_code()` with a
  `on_stdout` callback
- Always call `sandbox.kill()` in a `finally` block

### Error handling
- Wrap all external calls (DB, API, E2B) in try/except with informative messages
- Use `tenacity` for retries on transient failures (max 3 attempts,
  exponential back-off)
- Log with the stdlib `logging` module — no `print()` statements in library code

---

## Data flow

### Ingestion (`scripts/ingest.py`)

```
PDF / TXT / MD files
  → splitter.load_and_split()          # LangChain document loaders + RecursiveCharacterTextSplitter
  → embedder.embed_documents()         # batch embed chunks
  → vector_store.add_documents()       # upsert into pgvector with metadata
```

Each stored document chunk carries this metadata:
```python
{
  "source": "filename.pdf",
  "page": 3,                 # page number if applicable
  "chunk_index": 12,
  "ingested_at": "2025-04-16T10:00:00Z"
}
```

### Query (`scripts/query.py`)

```
user_question (str)
  → embedder.embed_query()             # single embedding call
  → vector_store.similarity_search()  # top-k cosine neighbours
  → rag_chain.invoke()                 # assemble prompt + call HuggingFace LLM
  → print answer + sources
```

---

## Running locally (without E2B)

```bash
# 1. Start PostgreSQL with pgvector
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env && $EDITOR .env

# 4. Ingest documents
python scripts/ingest.py --docs-dir ./docs

# 5. Ask a question
python scripts/query.py --question "What is our refund policy?"
```

## Running in E2B sandbox

```bash
python scripts/sandbox_runner.py --mode ingest --docs-dir ./docs
python scripts/sandbox_runner.py --mode query --question "What is our refund policy?"
```

---

## Testing

```bash
pytest tests/ -v
```

Tests must not make real API calls. Use `pytest-mock` to patch:
- `langchain_huggingface.HuggingFaceEmbeddings`
- `langchain_huggingface.HuggingFaceEndpoint`
- `langchain_postgres.PGVector`
- `e2b.Sandbox`

---

## What Claude Code should NOT do

- Do not modify `.env` or `.env.example` with real credentials
- Do not use `langchain` < 0.2 imports (deprecated)
- Do not call `sys.exit()` inside library modules
- Do not write synchronous blocking code inside async contexts
- Do not store raw document text in pgvector metadata fields — only lightweight
  keys (source, page, chunk_index)
