# SmartKB

SmartKB is a retrieval-augmented question-answering system for local knowledge bases. It loads documents, splits them into chunks, embeds the chunks, stores vectors in Milvus, combines dense retrieval with BM25, and generates grounded answers with DeepSeek.

## Features

- Document loading for Markdown, text, PDF, and DOCX files.
- Recursive chunking with overlap for retrieval stability.
- Multi-backend embedding wrapper with Zhipu, DashScope, HuggingFace, and Google fallback options.
- Milvus vector storage with cosine similarity search.
- PostgreSQL metadata, chat history, and evaluation records.
- Hybrid retrieval with vector search, BM25, and reciprocal-rank fusion.
- LangGraph agent wrapper for route -> retrieve -> generate flow.
- Streamlit UI for document management, chat, system status, and evaluation.
- Import and end-to-end test scripts.

## Architecture

```text
Documents
  -> loader
  -> splitter
  -> embedding model
  -> Milvus vectors + PostgreSQL metadata
  -> hybrid retriever
  -> DeepSeek generation chain
  -> Streamlit UI / LangGraph agent
```

## Directory Structure

```text
smartkb-rag/
├── app.py
├── config.py
├── test_imports.py
├── test_e2e.py
├── agent/
├── database/
├── eval/
├── models/
├── rag/
├── documents/
└── docs/
```

## Quick Start

```powershell
cd smartkb-rag
E:\Anaconda_envs\envs\langchain\python.exe test_imports.py
E:\Anaconda_envs\envs\langchain\python.exe test_e2e.py
streamlit run app.py --server.port 8501
```

Open http://localhost:8501.

## Configuration

Copy `.env.example` to `.env` and fill in provider keys and database addresses for live runs. The default tests run offline and do not require real credentials. Do not commit real credentials.

## Documentation

- `docs/PROJECT_REPORT.md`: complete project report.
- `docs/ARCHITECTURE.md`: module responsibilities and data flow.
- `docs/EVALUATION.md`: evaluation metrics and test strategy.
- `docs/DEPLOYMENT.md`: local runbook and troubleshooting.
