"""
LangGraph RAG Agent
===================
基于 LangGraph 构建的智能 RAG Agent。
Agent 流程:
  Router → [Retrieve → Generate]
         → [List Documents]
         → [Direct Chat]

Agent 决策逻辑:
  1. 用户询问知识库内容 → 先检索再生成
  2. 用户问有哪些文档 → 列出文档列表
  3. 用户闲聊 → 直接对话
"""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from models.llm import get_llm
from rag.retriever import HybridRetriever
from rag.chain import build_context, RAG_SYSTEM_PROMPT


# ============================================================
# State 定义
# ============================================================

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str                       # 用户原始问题
    retrieved_docs: List[dict]       # 检索到的文档
    final_answer: str                # 最终答案
    sources: List[str]               # 引用来源
    route: str                       # 路由决策


# ============================================================
# 路由节点
# ============================================================

ROUTER_PROMPT = """你是一个智能路由器，负责判断用户问题的类型。

请根据用户的问题，选择以下路径之一:
- "retrieve": 用户询问关于特定知识、文档内容的问题，需要检索知识库
- "list": 用户想知道知识库中有哪些文档
- "chat": 用户在进行闲聊、问候、或询问与知识库无关的问题

只回答一个词: retrieve, list, 或 chat。"""


def router_node(state: AgentState) -> AgentState:
    """
    路由节点: 判断用户意图并决定后续流程
    """
    messages = state["messages"]
    query = messages[-1].content if messages else ""

    state["query"] = query

    # 简单关键词判断 + LLM 兜底
    list_keywords = ["有哪些文档", "文档列表", "知识库有什么", "上传了什么",
                      "what documents", "list files", "库里有"]
    if any(kw in query.lower() for kw in list_keywords):
        state["route"] = "list"
        return state

    # 如果知识库为空，直接聊天模式
    chat_keywords = ["你好", "谢谢", "再见", "hello", "hi", "你是谁",
                      "你能做什么", "帮助", "help"]
    if any(kw in query.lower() for kw in chat_keywords):
        state["route"] = "chat"
        return state

    # 默认: 检索模式（尝试从知识库回答）
    state["route"] = "retrieve"
    return state


# ============================================================
# 检索节点
# ============================================================

class RetrieveNode:
    """检索节点"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def __call__(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            results = self.retriever.search(query, top_k=5)
            state["retrieved_docs"] = results
        except Exception as e:
            print(f"[Agent] 检索失败: {e}")
            state["retrieved_docs"] = []
        return state


# ============================================================
# 生成节点
# ============================================================

class GenerateNode:
    """生成节点: 基于检索结果生成答案"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def __call__(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs = state.get("retrieved_docs", [])

        if not docs:
            state["final_answer"] = "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试换个问法，或先上传相关文档。"
            state["sources"] = []
            return state

        context = build_context(docs)
        user_message = f"""以下是与问题相关的文档资料：

{context}

---
基于以上资料，请回答用户的问题。

用户问题: {query}
"""

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.chat(messages)
            state["final_answer"] = response
        except Exception as e:
            state["final_answer"] = f"生成回答时出错: {e}"
            state["final_answer"] += "\n\n但已检索到相关文档片段，您可以查看以下来源。"

        state["sources"] = list(set(
            d.get("file_name", "未知") for d in docs
        ))
        return state


# ============================================================
# 列表节点
# ============================================================

class ListNode:
    """列出知识库文档"""

    def __call__(self, state: AgentState) -> AgentState:
        from database.postgres_store import get_postgres_store
        pg = get_postgres_store()
        docs = pg.get_all_documents()

        if not docs:
            state["final_answer"] = "📚 知识库中还没有文档。请先在侧边栏上传文档！"
        else:
            parts = ["📚 **知识库现有文档:**\n"]
            for doc in docs:
                size_kb = doc.file_size / 1024 if doc.file_size else 0
                parts.append(
                    f"- **{doc.file_name}** "
                    f"({doc.file_type}, {size_kb:.1f}KB, "
                    f"{doc.chunk_count} 个分块)"
                )
            parts.append(f"\n共 {len(docs)} 个文档")
            state["final_answer"] = "\n".join(parts)

        state["sources"] = []
        return state


# ============================================================
# 对话节点
# ============================================================

class ChatNode:
    """直接对话节点"""

    def __init__(self):
        self.llm = get_llm(temperature=0.7)

    def __call__(self, state: AgentState) -> AgentState:
        query = state["query"]
        messages = [
            SystemMessage(
                content="你是一个友好的智能助手。请用简洁的中文回答用户。"
            ),
            HumanMessage(content=query),
        ]
        try:
            response = self.llm.chat(messages)
            state["final_answer"] = response
        except Exception as e:
            state["final_answer"] = f"抱歉，我暂时无法回答: {e}"

        state["sources"] = []
        return state


# ============================================================
# 路由条件
# ============================================================

def route_condition(state: AgentState) -> Literal["retrieve", "list", "chat"]:
    """根据路由决策选择下一节点"""
    return state["route"]


# ============================================================
# 构建 Graph
# ============================================================

def build_rag_agent(
    retriever: HybridRetriever,
) -> StateGraph:
    """
    构建 RAG Agent Graph

    流程:
      START → Router → Retrieve → Generate → END
                    → List ────────────────→ END
                    → Chat ────────────────→ END
    """
    # 创建 Graph
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", RetrieveNode(retriever))
    workflow.add_node("generate", GenerateNode())
    workflow.add_node("list_docs", ListNode())
    workflow.add_node("chat", ChatNode())

    # 设置入口
    workflow.set_entry_point("router")

    # 条件路由
    workflow.add_conditional_edges(
        "router",
        route_condition,
        {
            "retrieve": "retrieve",
            "list": "list_docs",
            "chat": "chat",
        },
    )

    # retrieve → generate → END
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # list_docs → END
    workflow.add_edge("list_docs", END)

    # chat → END
    workflow.add_edge("chat", END)

    return workflow.compile()
