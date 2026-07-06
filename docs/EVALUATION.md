# Evaluation

## Import Test

```powershell
E:\Anaconda_envs\envs\langchain\python.exe test_imports.py
```

Validates module imports, configuration loading, document splitting, and offline BM25 retrieval without external services.

## End-to-End Test

```powershell
E:\Anaconda_envs\envs\langchain\python.exe test_e2e.py
```

Validates the offline RAG path with fake embeddings and an in-memory vector store:

- Document loading.
- Document chunking.
- Fake embedding generation.
- In-memory vector retrieval.
- BM25 indexing.
- RRF hybrid fusion.
- Context construction with sources and scores.

## Live Checks

```powershell
E:\Anaconda_envs\envs\langchain\python.exe test_imports.py --live
E:\Anaconda_envs\envs\langchain\python.exe test_e2e.py --live
```

Live checks verify configured embedding, LLM, Milvus, and PostgreSQL services.

## Metrics

| Metric | Purpose |
| --- | --- |
| Hit Rate | Checks whether expected evidence is retrieved in top-k |
| MRR | Measures ranking quality of expected evidence |
| LLM Judge | Scores answer grounding and usefulness |

## Demo Dataset

The `documents/` directory contains small Markdown samples for deterministic local testing. Users can upload additional documents through the UI.
