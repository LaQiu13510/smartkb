"""
SmartKB — 智能知识库问答系统
=============================
基于 RAG (Retrieval-Augmented Generation) 技术的企业级知识库问答平台。

技术栈:
  LLM:      DeepSeek Chat  (国内直连)
  Embedding: 智谱AI embedding-2 (1024维, 国内直连)
  向量库:    Milvus (COSINE 相似度)
  元数据:    PostgreSQL
  检索:      混合检索 (向量 + BM25 + RRF 融合)
  Agent:     LangGraph 意图路由

运行方式:
  conda activate langchain
  streamlit run app.py
"""

import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

# ---- Streamlit 页面配置 (必须是第一个 st 调用) ----
st.set_page_config(
    page_title="SmartKB — 智能知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SmartKB v3.0 — 基于 RAG 的智能知识库问答系统",
    },
)

# ---- 自定义 CSS ----
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 聊天消息样式 */
    .chat-source {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.5rem;
    }
    .chat-latency {
        font-size: 0.75rem;
        color: #aaa;
    }
    /* 状态卡片 */
    .status-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .status-ok {
        background: #e8f5e9;
        border-left: 3px solid #4caf50;
    }
    .status-warn {
        background: #fff3e0;
        border-left: 3px solid #ff9800;
    }
    .status-err {
        background: #ffebee;
        border-left: 3px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# ---- 导入项目模块 ----
from config import (
    DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, MILVUS_HOST, MILVUS_PORT,
)
from models.embedding import get_embedding_model
from models.llm import get_llm
from database.milvus_store import get_milvus_store
from database.postgres_store import get_postgres_store
from rag.loader import DocumentLoader
from rag.splitter import TextSplitter
from rag.retriever import HybridRetriever
from rag.chain import get_rag_chain
from eval.metrics import RAGEvaluator
from eval.dataset import BASE_QUESTIONS


# ============================================================
# 会话状态初始化
# ============================================================

def init_session():
    """初始化 Streamlit 会话状态"""
    defaults = {
        "messages": [],
        "session_id": uuid.uuid4().hex[:12],
        "system_ready": False,
        "milvus_ok": False,
        "postgres_ok": False,
        "embedding_ok": False,
        "llm_ok": False,
        "embedding_backend": "",
        "embedding_dim": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 系统初始化 (带缓存)
# ============================================================

@st.cache_resource(show_spinner=False)
def init_milvus() -> tuple[bool, str]:
    """初始化 Milvus 连接"""
    try:
        store = get_milvus_store()
        store.connect()
        store.create_collection()
        count = store.count()
        return True, f"连接成功 · {count} 个向量"
    except Exception as e:
        return False, str(e)[:100]


@st.cache_resource(show_spinner=False)
def init_postgres() -> tuple[bool, str]:
    """初始化 PostgreSQL 连接"""
    try:
        pg = get_postgres_store()
        pg.init_tables()
        docs = pg.get_all_documents()
        return True, f"连接成功 · {len(docs)} 个文档"
    except Exception as e:
        return False, str(e)[:100]


@st.cache_resource(show_spinner=False)
def init_embedding() -> tuple[bool, str, str, int]:
    """初始化 Embedding 模型"""
    try:
        emb = get_embedding_model()
        ok, msg = emb.test_connection()
        return ok, msg, emb.backend, emb.dimension
    except Exception as e:
        return False, str(e)[:150], "unknown", 0


@st.cache_resource(show_spinner=False)
def init_llm() -> tuple[bool, str]:
    """初始化 DeepSeek LLM"""
    try:
        llm = get_llm()
        return llm.test_connection()
    except Exception as e:
        return False, str(e)[:150]


def get_or_create_retriever() -> HybridRetriever:
    """获取或创建全局检索器（非缓存，每次获取最新数据）"""
    milvus_store = get_milvus_store()
    retriever = HybridRetriever(milvus_store=milvus_store, top_k=TOP_K_RETRIEVAL)
    _rebuild_bm25(retriever)
    return retriever


def _rebuild_bm25(retriever: HybridRetriever):
    """重建 BM25 索引"""
    try:
        milvus_store = get_milvus_store()
        collection = milvus_store.get_collection()
        collection.load()
        results = collection.query(
            expr="id != ''",
            output_fields=["id", "content", "file_name", "chunk_index", "source_page"],
            limit=10000,
        )
        if results:
            docs_for_bm25 = [
                {
                    "chunk_id": r.get("id", ""),
                    "content": r.get("content", ""),
                    "file_name": r.get("file_name", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "source_page": r.get("source_page", 0),
                }
                for r in results
            ]
            retriever.build_bm25_index(docs_for_bm25)
    except Exception:
        pass  # 知识库为空时静默处理


# ============================================================
# 侧边栏
# ============================================================

def render_sidebar():
    """渲染侧边栏: 状态面板 + 文档管理 + 参数设置"""
    with st.sidebar:
        st.markdown("## 📚 SmartKB")
        st.caption("智能知识库问答系统 v3.0")

        # ---- 系统状态 ----
        st.markdown("### 🔌 系统状态")

        # 初始化所有组件
        if not st.session_state["system_ready"]:
            with st.spinner("正在初始化系统组件..."):
                milvus_ok, milvus_msg = init_milvus()
                pg_ok, pg_msg = init_postgres()
                emb_ok, emb_msg, emb_backend, emb_dim = init_embedding()
                llm_ok, llm_msg = init_llm()

                st.session_state.update({
                    "milvus_ok": milvus_ok,
                    "postgres_ok": pg_ok,
                    "embedding_ok": emb_ok,
                    "llm_ok": llm_ok,
                    "embedding_backend": emb_backend,
                    "embedding_dim": emb_dim,
                    "system_ready": True,
                })

        _status_line("Milvus 向量库", st.session_state["milvus_ok"], init_milvus()[1])
        _status_line("PostgreSQL", st.session_state["postgres_ok"], init_postgres()[1])
        _status_line("Embedding", st.session_state["embedding_ok"],
                     f"{st.session_state['embedding_backend']} · {st.session_state['embedding_dim']}维")
        _status_line("DeepSeek LLM", st.session_state["llm_ok"], "deepseek-chat")

        st.markdown("---")

        # ---- 文档上传 ----
        st.markdown("### 📤 上传文档")
        uploaded_files = st.file_uploader(
            "支持 PDF · TXT · MD · DOCX",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="上传到知识库后，系统会自动分块、向量化并建立索引",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 处理文档", type="primary", use_container_width=True,
                         disabled=not uploaded_files or not st.session_state["milvus_ok"]):
                _process_documents(uploaded_files)

        with col2:
            load_sample = st.button("📦 加载示例", use_container_width=True,
                                    help="加载内置示例文档用于测试",
                                    disabled=not st.session_state["milvus_ok"])
            if load_sample:
                _load_sample_documents()

        st.markdown("---")

        # ---- 知识库概况 ----
        st.markdown("### 📊 知识库概况")
        _render_kb_stats()

        st.markdown("---")

        # ---- 检索设置 ----
        st.markdown("### ⚙️ 检索设置")
        top_k = st.slider(
            "Top-K 返回数量",
            1, 20, TOP_K_RETRIEVAL,
            help="检索时返回的最相关文档片段数量",
        )
        # 更新模块级变量
        import rag.retriever as retriever_mod
        retriever = get_or_create_retriever()
        retriever.top_k = top_k

        st.markdown("---")

        # ---- 管理 ----
        st.markdown("### 🛠️ 管理")
        if st.button("🗑️ 清空知识库", use_container_width=True):
            if st.session_state.get("_confirm_clear"):
                _clear_kb()
                st.session_state["_confirm_clear"] = False
                st.rerun()
            else:
                st.session_state["_confirm_clear"] = True
                st.warning("⚠️ 再次点击确认清空全部数据")
        if st.session_state.get("_confirm_clear"):
            st.caption("↑ 确认清空？此操作不可撤销")

        # ---- 底部 ----
        st.markdown("---")
        st.caption(
            f"Session: {st.session_state['session_id']}\n\n"
            "Powered by DeepSeek + 智谱AI + Milvus"
        )


def _status_line(label: str, ok: bool, detail: str):
    """渲染一行系统状态"""
    icon = "🟢" if ok else "🔴"
    st.caption(f"{icon} **{label}**: {detail}")


def _render_kb_stats():
    """渲染知识库统计信息"""
    try:
        milvus = get_milvus_store()
        pg = get_postgres_store()
        vector_count = milvus.count()
        docs = pg.get_all_documents()

        col_a, col_b = st.columns(2)
        col_a.metric("📄 文档", len(docs))
        col_b.metric("🧩 向量块", vector_count)

        if docs:
            total_chars = sum(d.total_chars or 0 for d in docs)
            st.caption(f"总字符: {total_chars:,}")

            st.markdown("**最近文档:**")
            for doc in docs[:5]:
                st.caption(f"📄 {doc.file_name}")
    except Exception as e:
        st.caption(f"获取信息失败: {e}")


# ============================================================
# 文档处理
# ============================================================

def _process_documents(uploaded_files):
    """完整的文档处理流水线"""
    if not uploaded_files:
        return

    progress = st.progress(0, "准备处理...")
    status = st.empty()

    # Phase 1: 解析文档
    all_chunks = []
    file_infos = []
    parse_errors = []

    for i, uf in enumerate(uploaded_files):
        status.text(f"📖 解析: {uf.name}")
        progress.progress((i + 0.5) / (len(uploaded_files) + 4), f"解析: {uf.name}")

        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        try:
            docs = DocumentLoader.load_file(tmp_path)
            for doc in docs:
                doc.metadata["original_filename"] = uf.name
            splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunk_list = splitter.split_documents(docs)
            all_chunks.extend(chunk_list)
            file_infos.append({
                "file_name": uf.name,
                "file_type": suffix.replace(".", ""),
                "file_size": uf.size,
                "chunk_count": len(chunk_list),
                "total_chars": sum(len(c.page_content) for c in chunk_list),
            })
        except Exception as e:
            parse_errors.append(f"{uf.name}: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if parse_errors:
        for err in parse_errors:
            st.warning(f"⚠️ 解析失败: {err}")

    if not all_chunks:
        st.error("没有可处理的文档内容")
        return

    status.text(f"📊 共 {len(all_chunks)} 个文本块")

    # Phase 2: 向量化 (Embedding)
    progress.progress(0.6, "🧠 向量化...")
    status.text("🧠 正在生成向量嵌入...")

    embedding_model = get_embedding_model()
    contents = [c.page_content for c in all_chunks]
    all_embeddings = []

    batch_size = 40
    for j in range(0, len(contents), batch_size):
        batch = contents[j:j + batch_size]
        pct = 0.6 + 0.15 * (j / len(contents))
        progress.progress(pct, f"向量化: {j + 1}-{min(j + batch_size, len(contents))}/{len(contents)}")
        embeddings = embedding_model.embed_documents(batch)
        all_embeddings.extend(embeddings)

    # Phase 3: 存入 Milvus
    progress.progress(0.8, "💾 存储到向量库...")
    status.text("💾 正在存入 Milvus...")

    milvus = get_milvus_store()
    # 确保 Collection 维度正确 (重建如果维度不匹配)
    _ensure_collection_dim(milvus, len(all_embeddings[0]) if all_embeddings else 1024)

    ids, insert_contents, insert_embeddings, file_names, chunk_indices, source_pages = (
        [], [], [], [], [], []
    )

    for idx, (chunk, embedding) in enumerate(zip(all_chunks, all_embeddings)):
        meta = chunk.metadata
        chunk_id = hashlib.md5(
            (meta.get("original_filename", "") + str(idx)).encode()
        ).hexdigest()[:24]
        ids.append(chunk_id)
        insert_contents.append(chunk.page_content[:65535])
        insert_embeddings.append(embedding)
        file_names.append(meta.get("original_filename", meta.get("source", "")))
        chunk_indices.append(meta.get("chunk_index", idx))
        source_pages.append(meta.get("page", 0))

    try:
        count = milvus.insert(
            ids=ids, contents=insert_contents, embeddings=insert_embeddings,
            file_names=file_names, chunk_indices=chunk_indices, source_pages=source_pages,
        )
        st.success(f"✅ 已存储 {count} 条向量到 Milvus")
    except Exception as e:
        st.error(f"存储失败: {e}")
        return

    # Phase 4: 元数据存 PostgreSQL
    progress.progress(0.9, "📝 保存元数据...")
    status.text("📝 正在保存元数据...")
    pg = get_postgres_store()
    for info in file_infos:
        try:
            pg.add_document(**info)
        except Exception as e:
            st.warning(f"保存元数据失败 ({info['file_name']}): {e}")

    # Phase 5: 重建 BM25 索引
    progress.progress(0.95, "🔧 重建索引...")
    status.text("🔧 重建 BM25 索引...")
    _rebuild_bm25(get_or_create_retriever())

    progress.progress(1.0, "✅ 完成!")
    status.text("✅ 文档处理完成")

    # 清理缓存以刷新状态
    st.cache_resource.clear()
    st.session_state["system_ready"] = False
    st.rerun()


def _ensure_collection_dim(milvus, dim: int):
    """确保 Milvus Collection 维度匹配，不匹配则重建"""
    try:
        collection = milvus.get_collection()
        collection.load()
        # 检查现有 schema 的维度
        from pymilvus import Collection
        col = Collection(milvus._collection.name) if milvus._collection else None
        if col:
            for field in col.schema.fields:
                if field.name == "embedding":
                    if field.params.get("dim") != dim:
                        print(f"[App] 维度变化: {field.params.get('dim')} -> {dim}, 重建 Collection")
                        milvus.drop_collection()
                        milvus.create_collection()
                    break
    except Exception:
        pass


def _load_sample_documents():
    """加载内置示例文档"""
    sample_dir = Path(__file__).parent / "documents"
    sample_dir.mkdir(exist_ok=True)

    # 创建示例文档
    samples = {
        "RAG技术白皮书.md": """# RAG (检索增强生成) 技术白皮书

## 概述

RAG (Retrieval-Augmented Generation) 是一种结合了信息检索与文本生成的人工智能技术。它通过从外部知识库中检索相关文档片段，然后将这些片段作为上下文提供给大语言模型，从而生成更准确、更可靠的回答。

## 核心组件

### 1. 文档加载器 (Document Loader)
负责从多种数据源加载文档，支持的格式包括：
- PDF 文档
- Microsoft Word (.docx)
- 纯文本 (.txt)
- Markdown (.md)
- HTML 网页

### 2. 文本分割器 (Text Splitter)
将长文档切分为适当大小的文本块 (Chunk)。分割策略包括：
- **递归字符分割**: 按段落→句子→短语→字符的优先级逐级切分
- **语义分割**: 基于 Embedding 相似度边界检测
- **固定大小分割**: 按固定字符数切分，带重叠窗口

推荐参数: chunk_size=500, chunk_overlap=50

### 3. 嵌入模型 (Embedding Model)
将文本块转换为高维向量表示。常用模型：
- 智谱AI embedding-2: 1024维, 支持中英双语
- OpenAI text-embedding-3-small: 1536维
- Google text-embedding-004: 768维

### 4. 向量数据库 (Vector Database)
高效存储和检索向量。主流选择：
- **Milvus**: 开源分布式向量数据库, 支持多种索引算法
- Pinecone: 全托管向量数据库服务
- Chroma: 轻量级嵌入式向量数据库

Milvus 支持的索引类型:
- IVF_FLAT: 倒排索引 + 精确搜索, 平衡性能与精度
- HNSW: 分层可导航小世界图, 高性能近似搜索
- IVF_PQ: 乘积量化压缩, 适合大规模数据

### 5. 混合检索 (Hybrid Search)

单一的向量检索虽然能捕获语义相似性，但对精确关键词匹配能力较弱。
混合检索结合了两种方法的优势：

**向量检索** (Semantic Search):
- 使用 COSINE 相似度计算文本语义相似性
- 能理解同义词、近义词和相关概念
- 对模糊查询和自然语言问题效果好

**BM25 关键词检索**:
- 基于词频和逆文档频率的经典排序算法
- 对精确术语、编号、型号等关键词匹配效果好
- 速度快，可解释性强

**RRF 融合** (Reciprocal Rank Fusion):
```
RRF_score = w_vector / (k + rank_vector) + w_bm25 / (k + rank_bm25)
```
推荐参数: w_vector=0.6, w_bm25=0.4, k=60

## 性能指标

| 指标 | 纯向量检索 | 混合检索 | 提升 |
|------|----------|---------|------|
| Hit Rate | 65% | 92% | +41.5% |
| MRR | 0.52 | 0.84 | +61.5% |
| 关键词查询命中率 | 34% | 88% | +158% |

## 优化方向

1. **Cross-encoder Reranker**: 对检索结果进行精排
2. **Query Rewriting**: 查询改写和扩展
3. **HyDE**: 假设文档嵌入 (Hypothetical Document Embeddings)
4. **Multi-Agent**: 多智能体协作检索
""",

        "企业知识库运维指南.md": """# 企业知识库运维指南

## 系统架构

企业知识库采用微服务架构，包含以下核心服务：

### 基础设施
- **应用服务器**: 负责处理用户请求，运行 AI 推理服务
- **数据库集群**: PostgreSQL 主从架构，存储结构化元数据
- **向量数据库**: Milvus 集群，存储文档向量嵌入
- **缓存层**: Redis 集群，加速热点查询
- **消息队列**: RabbitMQ，异步任务处理

### 部署要求

| 组件 | CPU | 内存 | 磁盘 |
|------|-----|------|------|
| PostgreSQL | 4核 | 16GB | 500GB SSD |
| Milvus | 8核 | 32GB | 1TB SSD |
| Redis | 2核 | 8GB | 50GB SSD |
| 应用服务器 | 8核 | 32GB | 200GB SSD |

### 安全策略

1. **访问控制**: 基于 RBAC 的权限管理
2. **数据加密**: TLS 1.3 传输加密 + AES-256 存储加密
3. **审计日志**: 全量操作日志记录，保留 90 天
4. **备份策略**: 每日增量备份 + 每周全量备份

## 常见问题

### Q: 文档上传后检索不到？
检查步骤:
1. 确认文档解析成功 (查看文档状态)
2. 检查分块是否合理 (chunk_size 建议 500-800)
3. 验证 Embedding 服务可用
4. 检查 Milvus Collection 是否正常加载

### Q: 检索延迟高？
优化建议:
1. 调整 Milvus 索引参数 (nlist, nprobe)
2. 启用 Redis 缓存热点查询
3. 减少 Top-K 返回数量
4. 使用更轻量的 Embedding 模型

### Q: 答案不准确？
改进方向:
1. 优化 Prompt 模板
2. 引入 Reranker 精排
3. 增加文档覆盖范围
4. 调整 RRF 融合权重
""",

        "Python项目开发规范.md": """# Python 项目开发规范 v2.0

## 代码风格

遵循 PEP 8 规范，使用以下工具:
- **Ruff**: 代码格式化和 Lint (替代 Flake8 + isort + Black)
- **mypy**: 静态类型检查
- **pytest**: 单元测试框架

## 项目结构

```
project/
├── src/               # 源代码
│   ├── __init__.py
│   ├── models/        # 数据模型
│   ├── services/      # 业务逻辑
│   ├── api/           # API 路由
│   └── utils/         # 工具函数
├── tests/             # 测试代码
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/              # 文档
├── scripts/           # 脚本
├── pyproject.toml     # 项目配置
└── README.md
```

## 依赖管理

使用 `pyproject.toml` 管理依赖:
```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "pymilvus>=2.4.0",
]
```

## 测试规范

- 单元测试覆盖率 > 80%
- 集成测试覆盖核心 API
- E2E 测试覆盖关键用户流程
- 使用 fixtures 管理测试数据

## Git 工作流

1. `main`: 生产环境分支
2. `develop`: 开发分支
3. `feature/*`: 功能分支
4. `fix/*`: 修复分支

Commit 规范 (Conventional Commits):
- `feat: 添加新功能`
- `fix: 修复某个Bug`
- `docs: 更新文档`
- `refactor: 重构代码`
""",
    }

    for filename, content in samples.items():
        filepath = sample_dir / filename
        filepath.write_text(content, encoding="utf-8")

    # 用 DocumentLoader 处理
    all_chunks = []
    file_infos = []

    for filename in samples:
        filepath = sample_dir / filename
        try:
            docs = DocumentLoader.load_file(str(filepath))
            splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunk_list = splitter.split_documents(docs)
            for c in chunk_list:
                c.metadata["original_filename"] = filename
            all_chunks.extend(chunk_list)
            file_infos.append({
                "file_name": filename,
                "file_type": "md",
                "file_size": filepath.stat().st_size,
                "chunk_count": len(chunk_list),
                "total_chars": sum(len(c.page_content) for c in chunk_list),
            })
        except Exception as e:
            st.error(f"加载示例文件失败 {filename}: {e}")
            return

    st.info(f"📦 正在处理 {len(samples)} 个示例文档 ({len(all_chunks)} 个文本块)...")

    # Embedding
    emb = get_embedding_model()
    contents = [c.page_content for c in all_chunks]
    embeddings = []
    for j in range(0, len(contents), 40):
        batch = contents[j:j + 40]
        embeddings.extend(emb.embed_documents(batch))

    # 存入 Milvus
    milvus = get_milvus_store()
    _ensure_collection_dim(milvus, len(embeddings[0]) if embeddings else emb.dimension)

    ids, ins_c, ins_e, fns, cis, sps = [], [], [], [], [], []
    for idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        meta = chunk.metadata
        ids.append(hashlib.md5((meta.get("original_filename", "") + str(idx)).encode()).hexdigest()[:24])
        ins_c.append(chunk.page_content[:65535])
        ins_e.append(embedding)
        fns.append(meta.get("original_filename", ""))
        cis.append(meta.get("chunk_index", idx))
        sps.append(meta.get("page", 0))

    milvus.insert(ids=ids, contents=ins_c, embeddings=ins_e, file_names=fns,
                   chunk_indices=cis, source_pages=sps)

    # PostgreSQL
    pg = get_postgres_store()
    for info in file_infos:
        pg.add_document(**info)

    _rebuild_bm25(get_or_create_retriever())
    st.cache_resource.clear()
    st.session_state["system_ready"] = False
    st.success(f"✅ 已加载 {len(samples)} 个示例文档 ({len(all_chunks)} 个向量块)")
    st.rerun()


def _clear_kb():
    """清空知识库"""
    try:
        milvus = get_milvus_store()
        milvus.drop_collection()
        milvus.create_collection()
        st.cache_resource.clear()
        st.session_state["system_ready"] = False
        st.success("✅ 知识库已清空")
    except Exception as e:
        st.error(f"清空失败: {e}")


# ============================================================
# 对话主界面
# ============================================================

def render_chat():
    """渲染对话主界面"""
    st.markdown("## 💬 智能问答")
    st.caption("基于混合检索 (向量+BM25+RRF) 的知识库问答 — 上传文档后开始提问")

    if not st.session_state.get("milvus_ok", False):
        st.warning(
            f"Milvus 向量库当前不可用，请先启动服务并确认地址 "
            f"{MILVUS_HOST}:{MILVUS_PORT} 可连接。"
        )
        return

    # 欢迎提示
    if not st.session_state["messages"]:
        milvus = get_milvus_store()
        if milvus.count() == 0:
            st.info(
                "👋 **欢迎使用 SmartKB！**\n\n"
                "目前知识库为空，请先:\n"
                "1. 点击侧边栏「📦 加载示例」载入示例文档\n"
                "2. 或上传你自己的 PDF/TXT/MD/DOCX 文档\n\n"
                "然后就可以用自然语言提问了！"
            )
        else:
            st.info(
                f"👋 知识库就绪！已加载 **{milvus.count()}** 个文档片段，开始提问吧～\n\n"
                "试试: 「RAG有哪些核心组件？」「企业部署需要什么配置？」「Python项目规范是什么？」"
            )

    # 渲染历史消息
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {src}")
            if msg.get("latency_ms"):
                st.caption(f"⏱️ {msg['latency_ms']:.0f}ms")

    # 输入框
    if prompt := st.chat_input("请输入您的问题...", key="chat_main"):
        _handle_chat(prompt)


def _handle_chat(prompt: str):
    """处理一轮对话"""
    # 添加用户消息
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()

        with st.spinner("🤔 正在检索并生成答案..."):
            try:
                response = _rag_query(prompt)
                msg_placeholder.markdown(response["answer"])

                if response.get("sources"):
                    with st.expander("📎 参考来源"):
                        for src in response["sources"]:
                            st.caption(f"📄 {src}")

                st.caption(
                    f"⏱️ 检索: {response.get('retrieval_latency_ms', 0):.0f}ms "
                    f" | 生成: {response.get('generation_latency_ms', 0):.0f}ms "
                    f" | 共检索 {response.get('retrieved_count', 0)} 个片段"
                )
            except Exception as e:
                msg_placeholder.error(f"处理失败: {e}")
                response = {"answer": f"出错了: {e}", "sources": [], "latency_ms": 0}

    # 保存到会话
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response.get("sources", []),
        "latency_ms": response.get("latency_ms", 0),
    })

    # 异步保存到 PostgreSQL
    try:
        pg = get_postgres_store()
        pg.add_chat(
            session_id=st.session_state["session_id"],
            role="user", content=prompt,
        )
        pg.add_chat(
            session_id=st.session_state["session_id"],
            role="assistant", content=response["answer"],
            sources=json.dumps(response.get("sources", []), ensure_ascii=False),
            latency_ms=response.get("latency_ms", 0),
        )
    except Exception:
        pass


def _rag_query(query: str) -> dict:
    """完整的 RAG 查询流程"""
    total_start = time.time()

    # 1. 检索
    t0 = time.time()
    retriever = get_or_create_retriever()
    results = retriever.search(query, top_k=TOP_K_RETRIEVAL)
    retrieval_ms = (time.time() - t0) * 1000

    # 2. 生成
    rag_chain = get_rag_chain()
    gen_result = rag_chain.answer(
        query=query,
        retrieval_results=results,
        chat_history=st.session_state.get("messages", [])[-6:],
    )

    total_ms = (time.time() - total_start) * 1000

    return {
        "answer": gen_result["answer"],
        "sources": gen_result["sources"],
        "retrieved_count": len(results),
        "retrieval_latency_ms": round(retrieval_ms, 1),
        "generation_latency_ms": gen_result["latency_ms"],
        "latency_ms": round(total_ms, 1),
    }


# ============================================================
# 评测页面
# ============================================================

def render_evaluation():
    """渲染系统评测页面"""
    st.markdown("## 📊 系统评测")
    st.caption("自动化评测 RAG 系统的检索和生成质量")

    if not st.session_state.get("milvus_ok", False):
        st.warning(
            f"Milvus 向量库当前不可用，请先启动服务并确认地址 "
            f"{MILVUS_HOST}:{MILVUS_PORT} 可连接。"
        )
        return

    milvus = get_milvus_store()
    vector_count = milvus.count()

    if vector_count == 0:
        st.warning("⚠️ 知识库为空，请先上传文档或加载示例")
        return

    st.info(f"📊 当前知识库: {vector_count} 个向量块 · Embedding: {st.session_state['embedding_backend']} · {st.session_state['embedding_dim']}维")

    # 评测问题
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🔬 评测问题集")
        custom_q = st.text_area(
            "每行一个问题 (留空使用默认问题集):",
            height=120,
            placeholder="RAG技术的核心组件有哪些？\n企业部署需要什么硬件配置？",
        )
    with col2:
        st.markdown("### 📈 历史评测")
        try:
            pg = get_postgres_store()
            prev_stats = pg.get_evaluation_stats()
            if prev_stats["total"] > 0:
                st.metric("历史评测次数", prev_stats["total"])
                st.metric("平均 Hit Rate", f"{prev_stats['avg_hit_rate']*100:.1f}%")
                st.metric("平均 MRR", f"{prev_stats['avg_mrr']:.4f}")
                st.metric("平均延迟", f"{prev_stats['avg_latency_ms']:.0f}ms")
            else:
                st.caption("暂无历史评测数据")
        except Exception:
            st.caption("无法连接数据库")

    if st.button("🔄 开始评测", type="primary", use_container_width=True):
        questions = (
            [{"question": q.strip(), "keywords": []}
             for q in custom_q.strip().split("\n") if q.strip()]
            if custom_q.strip()
            else BASE_QUESTIONS
        )

        st.markdown(f"---\n### 📋 正在评测 {len(questions)} 个问题...")

        with st.spinner("评测进行中..."):
            retriever = get_or_create_retriever()
            evaluator = RAGEvaluator(retriever)
            report = evaluator.evaluate_batch(questions, verbose=False)

        # === 汇总指标 ===
        st.markdown("### 📈 评测结果")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Hit Rate", f"{report['hit_rate']*100:.1f}%")
        c2.metric("MRR", f"{report['avg_mrr']:.4f}")
        c3.metric("平均延迟", f"{report['avg_latency_ms']:.0f}ms")
        c4.metric("LLM 评判", f"{report['avg_judge_score']:.2f}/1.0")
        c5.metric("平均检索数", f"{report['avg_retrieved_count']:.1f}")

        # === 详细结果 ===
        st.markdown("### 📋 详细结果")
        for i, detail in enumerate(report["details"]):
            with st.expander(
                f"Q{i+1}: {detail['question'][:60]}..."
                f" | Hit: {detail['hit']} | MRR: {detail['mrr']:.4f} | "
                f"评分: {detail['judge_score']:.2f}"
            ):
                st.markdown(f"**问题:** {detail['question']}")
                st.markdown(f"**答案:** {detail['answer'][:600]}")
                st.caption(
                    f"来源: {', '.join(detail['sources']) if detail['sources'] else '无'} | "
                    f"检索: {detail['retrieval_latency_ms']:.0f}ms | "
                    f"生成: {detail['generation_latency_ms']:.0f}ms"
                )

        # 保存结果
        try:
            pg = get_postgres_store()
            for detail in report["details"]:
                pg.add_evaluation(
                    question=detail["question"],
                    generated_answer=detail["answer"],
                    hit_rate=detail["hit"],
                    mrr=detail["mrr"],
                    latency_ms=detail["total_latency_ms"],
                )
            st.success("✅ 评测结果已保存到数据库")
        except Exception as e:
            st.warning(f"保存评测结果失败: {e}")


# ============================================================
# 主入口
# ============================================================

def main():
    init_session()

    # 页面导航
    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "📋 导航",
            ["💬 智能问答", "📊 系统评测"],
            label_visibility="collapsed",
        )

    render_sidebar()

    if page == "💬 智能问答":
        render_chat()
    else:
        render_evaluation()


if __name__ == "__main__":
    main()


