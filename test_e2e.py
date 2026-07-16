# -*- coding: utf-8 -*-
"""SmartKB offline end-to-end test.

The default test runs without API keys, Milvus, or PostgreSQL. It validates the
RAG data path shape with fake embeddings and an in-memory vector store.
Use ``--live`` for external service checks.
"""

from __future__ import annotations

import hashlib
import io
import math
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))


TERMS = ["rag", "检索", "rrf", "部署", "milvus", "postgresql", "bm25", "文档"]


class FakeEmbeddingModel:
    backend = "fake"
    model_name = "offline-count-vector"
    dimension = len(TERMS)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [float(lower.count(term)) for term in TERMS]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


class InMemoryMilvus:
    def __init__(self):
        self.rows: list[dict] = []

    def insert(
        self,
        ids,
        contents,
        embeddings,
        file_names,
        chunk_indices,
        source_pages=None,
    ):
        source_pages = source_pages or [0] * len(ids)
        self.rows = []
        for idx, chunk_id in enumerate(ids):
            self.rows.append(
                {
                    "chunk_id": chunk_id,
                    "content": contents[idx],
                    "embedding": embeddings[idx],
                    "file_name": file_names[idx],
                    "chunk_index": chunk_indices[idx],
                    "source_page": source_pages[idx],
                }
            )
        return len(self.rows)

    def search(self, query_embedding, top_k=5):
        scored = []
        for row in self.rows:
            score = cosine(query_embedding, row["embedding"])
            scored.append({**row, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return [
            {
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "file_name": item["file_name"],
                "chunk_index": item["chunk_index"],
                "source_page": item["source_page"],
                "score": item["score"],
            }
            for item in scored[:top_k]
        ]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def run_offline_e2e():
    import rag.retriever as retriever_module
    import rag.splitter as splitter_module
    from rag.chain import build_context
    from rag.loader import DocumentLoader
    from rag.retriever import HybridRetriever
    from rag.splitter import TextSplitter

    print("=" * 60)
    print("SmartKB offline E2E")
    print("=" * 60)

    fake_embedding = FakeEmbeddingModel()
    retriever_module.get_embedding_model = lambda: fake_embedding
    splitter_module.get_embedding_model = lambda: fake_embedding

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "rag.md").write_text(
            "# RAG Guide\nRAG 使用文档加载、Embedding、Milvus、BM25 和 RRF 混合检索。",
            encoding="utf-8",
        )
        (tmp_path / "deploy.md").write_text(
            "# Deploy\n部署 SmartKB 需要 PostgreSQL、Milvus 和模型 API Key。",
            encoding="utf-8",
        )

        docs = DocumentLoader.load_directory(tmp_path)
        chunks = TextSplitter(chunk_size=80, chunk_overlap=10).split_documents(docs)
        assert len(docs) == 2
        assert chunks
        print(f"[OK] loaded={len(docs)} chunks={len(chunks)}")

        contents = [chunk.page_content for chunk in chunks]
        embeddings = fake_embedding.embed_documents(contents)

        ids = [
            hashlib.md5(f"{chunk.metadata.get('source')}:{idx}".encode()).hexdigest()[:24]
            for idx, chunk in enumerate(chunks)
        ]
        store = InMemoryMilvus()
        inserted = store.insert(
            ids=ids,
            contents=contents,
            embeddings=embeddings,
            file_names=[chunk.metadata.get("source", "unknown") for chunk in chunks],
            chunk_indices=[chunk.metadata.get("chunk_index", idx) for idx, chunk in enumerate(chunks)],
            source_pages=[chunk.metadata.get("page", 0) for chunk in chunks],
        )
        assert inserted == len(chunks)
        print(f"[OK] vectors={inserted} dim={fake_embedding.dimension}")

        bm25_docs = [
            {
                "chunk_id": ids[idx],
                "content": chunk.page_content,
                "file_name": chunk.metadata.get("source", ""),
                "chunk_index": chunk.metadata.get("chunk_index", idx),
                "source_page": chunk.metadata.get("page", 0),
            }
            for idx, chunk in enumerate(chunks)
        ]
        retriever = HybridRetriever(store, top_k=3)
        retriever.build_bm25_index(bm25_docs)

        rag_results = retriever.search("RAG 如何使用 BM25 和 RRF 混合检索？", top_k=2)
        deploy_results = retriever.search("部署需要 PostgreSQL 和 Milvus 吗？", top_k=2)
        assert rag_results and rag_results[0]["file_name"] == "rag.md"
        assert deploy_results and deploy_results[0]["file_name"] == "deploy.md"
        print(f"[OK] rag_top={rag_results[0]['file_name']} deploy_top={deploy_results[0]['file_name']}")

        context = build_context(rag_results)
        assert "来源: rag.md" in context
        assert "相关度" in context
        print("[OK] context builder includes sources and scores")

    print("\nSmartKB offline E2E passed.")


def run_live_health():
    from test_imports import live_checks

    live_checks()


def main():
    run_offline_e2e()
    if "--live" in sys.argv:
        print("\n[Live checks]")
        run_live_health()


if __name__ == "__main__":
    main()
