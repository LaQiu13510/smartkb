"""
Agent 工具定义
==============
为 LangGraph Agent 定义可调用的工具函数。
包括: 知识库检索、文档列表查询、对话历史查询等。
"""

from typing import List

from langchain_core.tools import tool

from rag.retriever import HybridRetriever
from rag.chain import get_rag_chain
from database.postgres_store import get_postgres_store


# 全局检索器引用 (在 graph 构建时注入)
_retriever: HybridRetriever | None = None


def set_retriever(retriever: HybridRetriever):
    global _retriever
    _retriever = retriever


@tool
def search_knowledge_base(query: str) -> str:
    """
    在知识库中搜索与查询相关的文档内容。
    当用户询问关于已上传文档的问题时，始终优先使用此工具。

    Args:
        query: 用户的查询问题或关键词

    Returns:
        检索到的相关文档内容和来源信息
    """
    if _retriever is None:
        return "错误: 检索器未初始化，请先上传文档建立知识库"

    results = _retriever.search(query, top_k=5)

    if not results:
        return "未找到相关文档内容"

    # 构建简洁的返回格式
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] 来源: {r.get('file_name', '未知')} | "
            f"相关度: {r.get('hybrid_score', r.get('score', 0)):.4f}\n"
            f"{r.get('content', '')[:500]}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def list_documents() -> str:
    """
    列出知识库中所有已上传的文档及其基本信息。
    当用户询问知识库中有哪些文档时使用此工具。
    """
    pg = get_postgres_store()
    docs = pg.get_all_documents()

    if not docs:
        return "知识库中暂无文档"

    parts = ["当前知识库文档列表:"]
    for doc in docs:
        size_kb = doc.file_size / 1024 if doc.file_size else 0
        parts.append(
            f"  - {doc.file_name} "
            f"({doc.file_type}, {size_kb:.1f}KB, "
            f"{doc.chunk_count} 个分块, "
            f"上传于 {doc.created_at.strftime('%Y-%m-%d %H:%M')})"
        )
    return "\n".join(parts)


@tool
def get_chat_summary() -> str:
    """
    获取当前对话的简要总结。
    当用户要求总结之前的对话内容时使用。
    """
    return "此功能正在开发中，当前暂不支持对话总结。"
