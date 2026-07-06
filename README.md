# SmartKB

SmartKB is a local retrieval-augmented generation (RAG) knowledge-base application. It ingests documents, chunks text, creates embeddings, stores vectors in Milvus, combines vector retrieval with BM25 keyword retrieval, and generates grounded answers with source-aware context.

## Features

- Load Markdown, TXT, PDF, and DOCX documents.
- Split documents with Chinese-friendly recursive text splitting.
- Support multiple embedding backends, including Zhipu, DashScope, HuggingFace, and Google.
- Store vectors in Milvus and metadata, chat history, and evaluation records in PostgreSQL.
- Retrieve with vector search, BM25 search, reciprocal-rank fusion, lightweight query rewriting, and reranking.
- Manage RAG context with source labels, deduplication, length budgeting, and sensitive-data redaction.
- Provide a LangGraph RAG agent with `retrieve`, `list`, and `chat` routes.
- Include a Streamlit interface for document management, chat, status checks, and evaluation.

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
  -> Streamlit UI / LangGraph agent
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

Do not commit real credentials.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py --server.port 8501
```

Open the local URL shown by Streamlit, upload documents, build the knowledge base, and start asking questions.

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
