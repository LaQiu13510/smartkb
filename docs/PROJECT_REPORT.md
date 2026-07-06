# SmartKB Project Report

## Overview

SmartKB is a complete RAG application for document-based question answering. It connects document ingestion, chunking, embedding, vector search, keyword search, answer generation, evaluation, and UI presentation into one runnable system.

## Goals

1. Build a local knowledge-base question-answering system with source-grounded answers.
2. Support reliable retrieval over mixed Chinese and English documents.
3. Store vectors and metadata in production-style infrastructure.
4. Provide evaluation metrics that reveal retrieval quality.
5. Offer a Streamlit UI suitable for local demonstration and debugging.

## Data Flow

```text
Upload document
  -> parse content
  -> split into chunks
  -> embed chunks
  -> write vectors to Milvus
  -> write metadata to PostgreSQL
  -> retrieve with vector search and BM25
  -> fuse rankings with RRF
  -> generate answer with DeepSeek
  -> show answer and sources
```

## Core Modules

| Module | Responsibility |
| --- | --- |
| `models/embedding.py` | Embedding backend selection and vector generation |
| `models/llm.py` | DeepSeek chat wrapper |
| `database/milvus_store.py` | Vector collection management and similarity search |
| `database/postgres_store.py` | Document metadata, chat history, and evaluation records |
| `rag/loader.py` | Document parsing |
| `rag/splitter.py` | Chunking strategy |
| `rag/retriever.py` | Hybrid retrieval and RRF fusion |
| `rag/chain.py` | Prompt assembly and answer generation |
| `agent/graph.py` | LangGraph flow around retrieval and generation |
| `eval/metrics.py` | Hit Rate, MRR, and LLM-based answer checks |
| `app.py` | Streamlit UI |

## Retrieval Design

SmartKB combines two retrieval strategies:

- Dense retrieval: embeds the query and searches Milvus by cosine similarity.
- BM25 retrieval: tokenizes text with jieba and ranks chunks by lexical relevance.

The final ranking uses reciprocal-rank fusion. This improves robustness when a query contains either semantic paraphrases or exact keywords.

## Embedding Design

The embedding wrapper attempts providers in a stable order. The primary provider uses a 1024-dimensional embedding model, and Milvus collection dimension checks prevent mismatched vectors from silently corrupting search results.

## Evaluation

The evaluation layer checks retrieval and answer quality with:

- Hit Rate: whether expected evidence appears in top-k results.
- MRR: how early the expected evidence appears.
- LLM judge: whether the generated answer is grounded and useful.

## UI

The Streamlit app provides:

- System health status.
- Document upload and indexing.
- Knowledge-base chat.
- Source display.
- Evaluation dashboard.
- Example document loading.

## Current Status

SmartKB is complete as a local RAG application:

- Core modules import successfully.
- Offline tests validate imports, document processing, BM25 retrieval, RRF fusion, and context construction without external services.
- Optional live checks validate embedding, LLM, Milvus, PostgreSQL, retrieval, generation, agent flow, and evaluation on a configured local environment.
- The UI can be started with Streamlit.
- GitHub-safe documentation and configuration templates are included.
