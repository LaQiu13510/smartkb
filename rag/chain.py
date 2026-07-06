"""
RAG 生成链
==========
将检索到的文档片段与用户问题拼接，通过 LLM 生成最终答案。
包含 Prompt 模板、上下文构建、来源追溯。
"""

import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument

from models.llm import get_llm
from rag.context import get_rag_context_manager


# ============================================================
# Prompt 模板
# ============================================================

RAG_SYSTEM_PROMPT = """你是一个智能知识库助手，基于给定的文档资料回答用户问题。

请严格遵循以下规则：
1. 只根据提供的文档内容作答，不要使用外部知识
2. 如果文档中没有相关信息，请如实回答"根据已有资料，无法回答此问题"
3. 回答时引用具体的来源文档，格式为【来源: 文件名】
4. 回答要简洁准确，优先使用中文
5. 如果文档内容相互矛盾，指出矛盾并列出不同观点
6. 在回答末尾列出引用的文件列表"""


def build_context(results: List[dict], max_chars: int = 4000) -> str:
    """
    将检索结果构建为 LLM 上下文

    策略：
    - 按 hybrid_score 降序排列
    - 填充到 max_chars 限制
    - 每条结果标注来源文件和页码
    """
    return get_rag_context_manager().build(results, max_chars=max_chars)


def build_user_message(query: str, context: str) -> str:
    """构建包含上下文和用户问题的完整消息"""
    return f"""以下是与问题相关的文档资料：

{context}

---
基于以上资料，请回答用户的问题。

用户问题: {query}

请给出详细、准确的回答。"""


class RAGChain:
    """RAG 问答链"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def answer(
        self,
        query: str,
        retrieval_results: List[dict],
        chat_history: List[dict] | None = None,
        stream: bool = False,
    ) -> dict:
        """
        基于检索结果生成答案

        Args:
            query: 用户问题
            retrieval_results: 检索结果列表
            chat_history: 历史对话 (可选)
            stream: 是否流式输出 (预留)

        Returns:
            {
                "answer": str,           # 生成的答案
                "sources": List[str],    # 引用来源列表
                "context": str,          # 使用的上下文
                "latency_ms": float,     # 响应延迟 (毫秒)
            }
        """
        start_time = time.time()

        # 构建上下文
        context = build_context(retrieval_results)

        # 构建消息
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

        # 如果有历史对话，加入最近的几轮
        if chat_history:
            for record in chat_history[-4:]:  # 最近 2 轮对话
                role = record.get("role", "user")
                content = record.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    # 用自定义类型避免 LangChain 类型检查问题
                    messages.append(SystemMessage(
                        content=f"[助手之前的回答]: {content[:200]}"
                    ))

        messages.append(HumanMessage(content=build_user_message(query, context)))

        # LLM 生成
        response = self.llm.chat(messages)

        latency = (time.time() - start_time) * 1000

        # 提取来源
        sources = list(set(
            r.get("file_name", "未知") for r in retrieval_results
        ))

        return {
            "answer": response,
            "sources": sources,
            "context": context,
            "latency_ms": round(latency, 1),
        }

    async def aanswer(
        self,
        query: str,
        retrieval_results: List[dict],
        chat_history: List[dict] | None = None,
    ) -> dict:
        """异步版本的答案生成"""
        start_time = time.time()

        # 构建上下文
        context = build_context(retrieval_results)

        # 构建消息
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

        if chat_history:
            for record in chat_history[-4:]:
                role = record.get("role", "user")
                content = record.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(
                        content=f"[助手之前的回答]: {content[:200]}"
                    ))

        messages.append(HumanMessage(content=build_user_message(query, context)))

        # 异步 LLM 生成
        response = await self.llm.achat(messages)

        latency = (time.time() - start_time) * 1000

        sources = list(set(
            r.get("file_name", "未知") for r in retrieval_results
        ))

        return {
            "answer": response,
            "sources": sources,
            "context": context,
            "latency_ms": round(latency, 1),
        }


# 全局单例
_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
