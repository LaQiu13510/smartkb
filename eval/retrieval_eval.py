"""Offline retrieval strategy evaluation for SmartKB."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.retriever as retriever_module
from rag.retriever import HybridRetriever


TERMS = ["rag", "检索", "rrf", "bm25", "部署", "milvus", "postgresql", "文档"]


class FakeEmbeddingModel:
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [float(lower.count(term)) for term in TERMS]


class InMemoryVectorStore:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def search(self, query_embedding, top_k: int = 5):
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


def build_retriever() -> HybridRetriever:
    embedding = FakeEmbeddingModel()
    docs = [
        {"chunk_id": "general", "file_name": "general.md", "content": "项目管理流程和会议纪要。", "chunk_index": 0, "source_page": 0},
        {"chunk_id": "rrf", "file_name": "rrf.md", "content": "Reciprocal Rank Fusion formula uses rank positions to fuse retrieval lists.", "chunk_index": 0, "source_page": 0},
        {"chunk_id": "rag", "file_name": "rag.md", "content": "RAG 检索增强生成包含文档加载、分块、Embedding、向量检索和答案生成。", "chunk_index": 0, "source_page": 0},
        {"chunk_id": "deploy", "file_name": "deploy.md", "content": "部署需要 PostgreSQL、Milvus、模型 API Key 和服务健康检查。", "chunk_index": 0, "source_page": 0},
    ]
    rows = [{**doc, "embedding": embedding.embed_query(doc["content"])} for doc in docs]
    retriever_module.get_embedding_model = lambda: embedding
    retriever = HybridRetriever(InMemoryVectorStore(rows), top_k=3)
    retriever.build_bm25_index(docs)
    return retriever


CASES = [
    {"query": "RAG 的核心组件有哪些？", "expected": "rag.md"},
    {"query": "reciprocal rank fusion formula", "expected": "rrf.md"},
    {"query": "PostgreSQL 和 Milvus 部署要求", "expected": "deploy.md"},
    {"query": "BM25 混合检索如何提升关键词查询？", "expected": "rag.md"},
]


def top_file(results: list[dict]) -> str:
    return results[0].get("file_name", "") if results else ""


def run_eval() -> dict:
    retriever = build_retriever()
    details = []
    scores = {"vector": 0, "bm25": 0, "hybrid": 0}

    for case in CASES:
        vector_top = top_file(retriever.search_vector_only(case["query"], top_k=3))
        bm25_top = top_file(retriever.search_bm25_only(case["query"], top_k=3))
        hybrid_top = top_file(retriever.search(case["query"], top_k=3))
        row = {
            "query": case["query"],
            "expected": case["expected"],
            "vector_top": vector_top,
            "bm25_top": bm25_top,
            "hybrid_top": hybrid_top,
        }
        for key, top in [("vector", vector_top), ("bm25", bm25_top), ("hybrid", hybrid_top)]:
            if top == case["expected"]:
                scores[key] += 1
        details.append(row)

    total = len(CASES)
    return {
        "project": "smartkb-rag",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "top1_accuracy": {
            key: round(value / total, 4)
            for key, value in scores.items()
        },
        "details": details,
    }


def main():
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
