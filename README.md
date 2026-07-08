# SmartKB

SmartKB is a local retrieval-augmented generation (RAG) knowledge-base application. It ingests documents, chunks text, creates embeddings, stores vectors in Milvus, combines vector retrieval with BM25 keyword retrieval, generates grounded answers with source-aware context, and provides a FastAPI web interface.

## Features

- Load Markdown, TXT, PDF, and DOCX documents.
- Split documents with semantic-aware chunking and recursive fallback.
- Support multiple embedding backends, including Zhipu, DashScope, HuggingFace, and Google.
- Store vectors in Milvus and metadata, chat history, and evaluation records in PostgreSQL.
- Retrieve with vector search, BM25 search, reciprocal-rank fusion, lightweight query rewriting, and reranking.
- Manage RAG context with source labels, deduplication, length budgeting, and sensitive-data redaction.
- Provide a LangGraph RAG agent with `retrieve`, `list`, and `chat` routes.
- Include a FastAPI dashboard for text indexing, knowledge-base chat, status checks, sources, and retrieved chunks.
- Stream answers to the browser with Server-Sent Events (SSE).

## Architecture

```text
Documents
  -> loader
  -> splitter
  -> embedding model
  -> Milvus vectors + PostgreSQL metadata
  -> hybrid retriever
  -> RAG context manager
  -> generation chain
  -> FastAPI UI / LangGraph agent
```

## Project Structure

```text
smartkb-rag/
├── app.py
├── config.py
├── test_imports.py
├── test_e2e.py
├── agent/
├── database/
├── docs/
├── documents/
├── eval/
├── models/
└── rag/
```

## Installation

```bash
git clone <your-repository-url>
cd smartkb-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your own service credentials.

```bash
cp .env.example .env
```

Required live services:

- DeepSeek-compatible chat API
- At least one embedding provider or a local HuggingFace embedding model
- Milvus
- PostgreSQL


## Usage

Run the FastAPI app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`, add text to the knowledge base, and start asking questions.

## Tests

The default tests are offline and do not require external services.

```bash
python test_imports.py
python test_e2e.py
python eval/agent_eval.py
python eval/retrieval_eval.py
```

Live checks can be run after `.env` is configured:

```bash
python test_imports.py --live
python test_e2e.py --live
```

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/DEPLOYMENT.md`
- `docs/EVALUATION.md`
- `docs/PROJECT_REPORT.md`
