"""SmartKB FastAPI、缓存、文件上传与 SSE 离线测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module
from cache_store import InMemoryTTLCacheStore


class FakeRetriever:
    def search(self, query: str, top_k: int = 3):
        return [
            {
                "chunk_id": "offline-1",
                "content": f"与 {query} 相关的离线知识片段。",
                "file_name": "offline.md",
                "chunk_index": 0,
                "source_page": 0,
                "hybrid_score": 0.91,
            }
        ][:top_k]


class FakeRAGChain:
    def answer(self, query: str, results: list[dict]):
        return {
            "answer": f"离线回答：{query}",
            "sources": [item["file_name"] for item in results],
            "latency_ms": 1.0,
        }

    def stream_answer(self, query: str, results: list[dict]):
        yield "离线流式"
        yield f"回答：{query}"


def patch_dependencies():
    cache = InMemoryTTLCacheStore("api-test")
    app_module.QUERY_CACHE_ENABLED = True
    app_module.get_cache_store = lambda: cache
    app_module.get_retriever_cached = lambda: FakeRetriever()
    app_module.get_rag_chain = lambda: FakeRAGChain()
    app_module.persist_chat = lambda *args, **kwargs: None
    return cache


def run_query_and_cache_checks(client: TestClient):
    payload = {"query": "RAG 如何工作？", "session_id": "api-test", "top_k": 3}
    first = client.post("/api/query", json=payload)
    assert first.status_code == 200, first.text
    assert first.json()["ok"] is True
    assert first.json()["cache_hit"] is False

    second = client.post("/api/query", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["cache_hit"] is True

    invalid = client.post("/api/query", json={**payload, "top_k": 99})
    assert invalid.status_code == 422
    print("[OK] query API, cache hit, and request validation")


def run_sse_checks(client: TestClient):
    response = client.get(
        "/api/query/stream",
        params={"query": "Milvus 有什么作用？", "session_id": "stream-test", "top_k": 3},
    )
    assert response.status_code == 200, response.text
    assert "event: stage" in response.text
    assert "event: delta" in response.text
    assert "离线流式" in response.text
    assert "event: final" in response.text

    invalid = client.get("/api/query/stream", params={"query": "", "top_k": 20})
    assert invalid.status_code == 422
    print("[OK] SSE token stream and query-parameter validation")


def run_upload_checks(client: TestClient):
    original = app_module.index_documents
    app_module.index_documents = lambda file_name, documents, file_size: {
        "ok": True,
        "file_name": file_name,
        "chunks": len(documents),
        "file_size": file_size,
    }
    try:
        response = client.post(
            "/api/index-file",
            files={"file": ("guide.txt", "SmartKB 文件上传测试".encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["ok"] is True
        assert response.json()["file_name"] == "guide.txt"

        unsupported = client.post(
            "/api/index-file",
            files={"file": ("unsafe.exe", b"binary", "application/octet-stream")},
        )
        assert unsupported.status_code == 200
        assert unsupported.json()["ok"] is False
    finally:
        app_module.index_documents = original
    print("[OK] supported upload and extension rejection")


def main():
    patch_dependencies()
    client = TestClient(app_module.app)
    run_query_and_cache_checks(client)
    run_sse_checks(client)
    run_upload_checks(client)
    print("\nSmartKB API checks passed.")


if __name__ == "__main__":
    main()
