# RAG Knowledge Assistant

An internal knowledge assistant that ingests documents (PDF, TXT, MD), stores embeddings in PostgreSQL with pgvector, and answers questions using Azure OpenAI.

## Architecture

- **LangChain** — orchestration (ingestion, retrieval, prompt assembly via LCEL)
- **pgvector** — PostgreSQL extension for vector similarity search
- **Azure OpenAI** — embeddings and chat completion (gpt-5-nano)
- **E2B** — optional sandboxed execution environment

## Quick Start

```bash
# 1. Start PostgreSQL with pgvector
docker-compose up -d

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 4. Ingest documents
python scripts/ingest.py --docs-dir ./docs

# 5. Ask a question
python scripts/query.py --question "What is our refund policy?"
```

## Project Structure

```
├── scripts/
│   ├── ingest.py              # Ingestion pipeline
│   ├── query.py               # Interactive query loop
│   └── sandbox_runner.py      # E2B sandbox wrapper
├── src/
│   ├── config.py              # Environment variable loading and validation
│   ├── embedder.py            # Azure OpenAI embeddings with caching
│   ├── splitter.py            # Document loading and text splitting
│   ├── vector_store.py        # pgvector connection and CRUD helpers
│   └── rag_chain.py           # LCEL RAG chain with Azure OpenAI
├── tests/                     # Unit tests (no real API calls)
├── test_walkthrough.ipynb     # Step-by-step Jupyter notebook
├── docker-compose.yml         # PostgreSQL + pgvector container
└── requirements.txt           # Python dependencies
```

## Environment Variables

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Base endpoint URL (e.g. `https://your-resource.cognitiveservices.azure.com/`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |
| `DATABASE_URL` | PostgreSQL connection string |
| `E2B_API_KEY` | E2B sandbox API key |

## Testing

```bash
pytest tests/ -v
```

Tests use mocks — no real API calls or database connections required.
