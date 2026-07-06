# SmartKB Architecture

## Layers

```text
UI Layer
  app.py

Agent Layer
  agent/graph.py
  agent/tools.py

RAG Layer
  rag/loader.py
  rag/splitter.py
  rag/retriever.py
  rag/chain.py

Storage Layer
  database/milvus_store.py
  database/postgres_store.py

Model Layer
  models/embedding.py
  models/llm.py

Evaluation Layer
  eval/metrics.py
  eval/dataset.py
```

## Document Processing

1. `DocumentLoader` extracts text from supported file types.
2. `TextSplitter` creates overlapping chunks.
3. `EmbeddingModel` converts chunks into vectors.
4. `MilvusStore` stores vectors and chunk payload.
5. `PostgresStore` stores document metadata and execution records.

## Retrieval

The retriever runs vector search and BM25 search, then fuses both result lists. This protects the system from two common failure modes:

- keyword-only matching missing semantic matches;
- vector-only matching missing exact terms, IDs, or rare names.

## Generation

The generation chain builds a grounded prompt from retrieved context and asks DeepSeek to answer with source awareness. The UI displays both the answer and supporting chunks.

## Agent Flow

The LangGraph wrapper keeps the flow explicit:

```text
router -> retrieve -> generate
```

This is intentionally compact. SmartKB focuses on the RAG pipeline, while AgentFlow focuses on multi-agent orchestration.
