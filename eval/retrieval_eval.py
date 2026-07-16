"""SmartKB 离线检索评测：Top-1、Recall@3、MRR@3 与 nDCG@3。"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.retriever as retriever_module
from rag.retriever import HybridRetriever


CONCEPTS = [
    ("rag", "retrieval augmented generation", "检索增强生成", "知识库问答"),
    ("rrf", "reciprocal rank fusion", "排名融合", "融合排序"),
    ("bm25", "关键词检索", "稀疏检索", "idf"),
    ("milvus", "向量数据库", "向量检索", "collection"),
    ("postgresql", "关系数据库", "元数据", "sql"),
    ("redis", "ttl", "热点缓存", "aof"),
    ("docker", "容器化", "镜像", "compose"),
    ("fastapi", "asgi", "接口服务", "api"),
    ("langgraph", "状态图", "supervisor", "worker"),
    ("mcp", "tool calling", "工具协议", "list_tools"),
    ("context", "上下文预算", "证据去重", "prompt"),
    ("memory", "长期记忆", "短期记忆", "会话历史"),
    ("evaluation", "mrr", "ndcg", "召回率"),
    ("security", "脱敏", "prompt injection", "敏感信息"),
    ("chunking", "语义分块", "文本切分", "chunk overlap"),
    ("deployment", "健康检查", "服务依赖", "启动顺序"),
    ("cache invalidation", "缓存失效", "索引刷新", "delete_prefix"),
    ("query rewrite", "查询改写", "查询扩展", "同义词"),
    ("rerank", "精排", "cross encoder", "重排序"),
    ("sse", "server sent events", "流式输出", "event stream"),
    ("checkpoint", "断点续跑", "任务恢复", "thread_id"),
    ("trace", "执行追踪", "可观察性", "agenttracestore"),
]


class FakeEmbeddingModel:
    model_name = "offline-concept-embedding"

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            float(sum(1 for alias in aliases if alias in lower))
            for aliases in CONCEPTS
        ]


class InMemoryVectorStore:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def search(self, query_embedding, top_k: int = 5):
        scored = []
        for row in self.rows:
            score = cosine(query_embedding, row["embedding"])
            if score > 0:
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


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


DOCUMENTS = [
    ("rag.md", "RAG 检索增强生成把知识库检索结果注入大模型，形成可追踪的企业知识库问答链路。"),
    ("rrf.md", "RRF 即 Reciprocal Rank Fusion，通过排名融合合并向量检索与关键词检索结果。"),
    ("bm25.md", "BM25 是关键词检索与稀疏检索算法，使用 TF、IDF 和文档长度计算相关性。"),
    ("milvus.md", "Milvus 是向量数据库，使用 collection、索引和余弦相似度完成向量检索。"),
    ("postgresql.md", "PostgreSQL 关系数据库保存文档元数据、会话记录和结构化 SQL 数据。"),
    ("redis.md", "Redis 使用 TTL 热点缓存降低重复查询延迟，并可通过 AOF 提供持久化。"),
    ("docker.md", "Docker 容器化把 FastAPI 服务打包为镜像，Compose 负责组织多个容器。"),
    ("fastapi.md", "FastAPI 是 ASGI 接口服务框架，提供类型校验、OpenAPI 和异步 API。"),
    ("langgraph.md", "LangGraph 使用状态图编排 Supervisor 与 Worker，支持条件路由和状态传递。"),
    ("mcp.md", "MCP 风格工具协议统一 list_tools、call_tool、输入 schema 与工具调用错误。"),
    ("context.md", "上下文治理通过上下文预算、证据去重、来源标注和 Prompt 裁剪控制输入。"),
    ("memory.md", "记忆系统用短期记忆保存会话历史，用长期记忆沉淀可复用任务经验。"),
    ("evaluation.md", "检索评测使用 Top-1、Recall@3、MRR 与 nDCG 衡量召回率和排序质量。"),
    ("security.md", "安全模块负责敏感信息脱敏、Prompt Injection 约束和工具调用次数限制。"),
    ("chunking.md", "语义分块结合 Embedding 边界、标题和段落进行文本切分，并保留 chunk overlap。"),
    ("deployment.md", "部署编排通过健康检查、服务依赖和启动顺序等待 PostgreSQL、Redis 与 Milvus。"),
    ("cache.md", "文档更新后调用 delete_prefix 完成缓存失效，并刷新 BM25 索引。"),
    ("rewrite.md", "Query Rewrite 查询改写通过查询扩展和同义词补充改善短问题召回。"),
    ("rerank.md", "Rerank 精排对扩大后的候选集执行重排序，可选 Cross Encoder 或轻量词汇覆盖度。"),
    ("sse.md", "SSE Server-Sent Events 使用 text/event-stream 与 X-Accel-Buffering 实现流式输出。"),
    ("checkpoint.md", "Checkpoint 使用 thread_id 保存图状态，支持任务恢复与断点续跑。"),
    ("trace.md", "AgentTraceStore 执行追踪记录路由、工具计划、观察结果、延迟和最终答案。"),
]


CASES = [
    ("RAG 的企业知识库问答链路是什么？", ["rag.md"]),
    ("retrieval augmented generation 如何工作？", ["rag.md"]),
    ("reciprocal rank fusion 的作用", ["rrf.md"]),
    ("RRF 怎样合并两路排名？", ["rrf.md"]),
    ("稀疏检索为什么需要 IDF？", ["bm25.md"]),
    ("BM25 如何处理专业关键词？", ["bm25.md"]),
    ("向量数据库 collection 如何组织数据？", ["milvus.md"]),
    ("Milvus 使用什么方式做相似度检索？", ["milvus.md"]),
    ("文档元数据应该保存在哪个关系数据库？", ["postgresql.md"]),
    ("SQL 会话记录如何持久化？", ["postgresql.md"]),
    ("TTL 热点缓存如何降低重复查询延迟？", ["redis.md"]),
    ("AOF 与缓存过期策略", ["redis.md"]),
    ("怎样用容器化镜像交付服务？", ["docker.md"]),
    ("Compose 如何组织多个容器？", ["docker.md", "deployment.md"]),
    ("ASGI 接口服务如何做参数校验？", ["fastapi.md"]),
    ("OpenAPI 文档由哪个 Web 框架提供？", ["fastapi.md"]),
    ("Supervisor 和 Worker 如何在状态图中协作？", ["langgraph.md"]),
    ("条件路由如何传递图状态？", ["langgraph.md"]),
    ("list_tools 和 call_tool 属于哪种工具协议？", ["mcp.md"]),
    ("工具输入 schema 为什么重要？", ["mcp.md"]),
    ("怎样控制 Prompt 的上下文预算并标注来源？", ["context.md"]),
    ("怎样避免回答引用互相重复的材料？", ["context.md"]),
    ("短期会话和长期任务经验如何分别保存？", ["memory.md"]),
    ("Recall@3、MRR 和 nDCG 分别衡量什么？", ["evaluation.md"]),
    ("如何防止密钥进入 Prompt？", ["security.md"]),
    ("Embedding 语义边界如何用于文本切分？", ["chunking.md"]),
    ("服务依赖和健康检查如何控制启动顺序？", ["deployment.md"]),
    ("文档更新后为什么要调用 delete_prefix？", ["cache.md"]),
    ("短问题如何通过同义词做查询扩展？", ["rewrite.md"]),
    ("Cross Encoder 应该在扩大候选集后做什么？", ["rerank.md"]),
    ("X-Accel-Buffering 在流式接口中有什么作用？", ["sse.md"]),
    ("thread_id 如何关联任务恢复状态？", ["checkpoint.md"]),
    ("服务重启后如何接着之前的步骤执行？", ["checkpoint.md"]),
    ("AgentTraceStore 会记录哪些执行信息？", ["trace.md"]),
]


def build_retriever() -> HybridRetriever:
    embedding = FakeEmbeddingModel()
    docs = [
        {
            "chunk_id": f"chunk-{index}",
            "file_name": file_name,
            "content": content,
            "chunk_index": 0,
            "source_page": 0,
        }
        for index, (file_name, content) in enumerate(DOCUMENTS)
    ]
    rows = [{**doc, "embedding": embedding.embed_query(doc["content"])} for doc in docs]
    retriever_module.get_embedding_model = lambda: embedding
    retriever = HybridRetriever(InMemoryVectorStore(rows), top_k=3)
    retriever.build_bm25_index(docs)
    return retriever


def ranked_files(results: list[dict]) -> list[str]:
    return list(dict.fromkeys(item.get("file_name", "") for item in results if item.get("file_name")))


def ranking_metrics(files: list[str], relevant: list[str], k: int = 3) -> dict[str, float]:
    top = files[:k]
    relevant_set = set(relevant)
    hits = [1 if file_name in relevant_set else 0 for file_name in top]
    top1 = float(bool(top) and top[0] in relevant_set)
    recall = sum(hits) / len(relevant_set) if relevant_set else 0.0
    first_rank = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
    ideal_hits = [1] * min(len(relevant_set), k)
    ideal_dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(ideal_hits, start=1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {"top1": top1, "recall_at_3": recall, "mrr_at_3": mrr, "ndcg_at_3": ndcg}


def run_eval() -> dict:
    retriever = build_retriever()
    totals = {
        method: {"top1": 0.0, "recall_at_3": 0.0, "mrr_at_3": 0.0, "ndcg_at_3": 0.0, "latency_ms": 0.0}
        for method in ("vector", "bm25", "hybrid")
    }
    details = []

    for query, relevant in CASES:
        method_results = {}
        calls = {
            "vector": lambda: retriever.search_vector_only(query, top_k=3),
            "bm25": lambda: retriever.search_bm25_only(query, top_k=3),
            "hybrid": lambda: retriever.search(query, top_k=3),
        }
        for method, call in calls.items():
            started = time.perf_counter()
            files = ranked_files(call())
            latency_ms = (time.perf_counter() - started) * 1000
            metrics = ranking_metrics(files, relevant)
            for name, value in metrics.items():
                totals[method][name] += value
            totals[method]["latency_ms"] += latency_ms
            method_results[method] = {"files": files, **metrics}
        details.append({"query": query, "relevant": relevant, "results": method_results})

    total = len(CASES)
    summary = {
        method: {
            name: round(value / total, 4 if name != "latency_ms" else 3)
            for name, value in values.items()
        }
        for method, values in totals.items()
    }
    return {
        "project": "smartkb-rag",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"documents": len(DOCUMENTS), "queries": total, "top_k": 3},
        "metrics": summary,
        "details": details,
    }


def main():
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
