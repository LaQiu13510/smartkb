# Deployment and Runbook

## Requirements

- Python 3.10+
- Milvus server
- PostgreSQL server
- DeepSeek-compatible chat API key
- At least one embedding provider key or a local embedding fallback

## Local Setup

```powershell
cd smartkb-rag
E:\Anaconda_envs\envs\langchain\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with local provider and storage settings.

## Run Tests

```powershell
E:\Anaconda_envs\envs\langchain\python.exe test_imports.py
E:\Anaconda_envs\envs\langchain\python.exe test_e2e.py
```

## Start UI

```powershell
streamlit run app.py --server.port 8501
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Embedding fails | Verify provider key and network access |
| Milvus collection dimension mismatch | Rebuild collection with the configured vector dimension |
| PostgreSQL write fails | Check `DB_URL`, user permissions, and network reachability |
| Retrieval returns no context | Confirm documents were indexed and BM25 index was rebuilt |
| UI starts but no answer appears | Run `test_e2e.py` to isolate model, storage, or retrieval failures |
