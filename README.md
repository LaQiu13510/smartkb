# SmartKB 企业知识库 RAG 系统

SmartKB 面向企业内部制度、技术规范、项目资料和运维手册分散，员工检索成本高的问题，构建从文档入库、混合检索、上下文治理到带来源回答的完整 RAG 链路，并提供可直接使用的 FastAPI Web 界面。

## 核心能力

- 支持 Markdown、TXT、PDF、DOCX 文档上传、解析、更新与删除。
- 使用标题、段落和 Embedding 相似度识别语义边界，Embedding 不可用时自动降级为词汇边界检测和递归字符切分。
- 使用 Milvus 完成向量检索，使用 BM25 完成关键词检索，并通过 RRF 融合两路排序。
- 支持轻量 Query Rewrite 和候选集 Rerank，可选 Cross Encoder 精排。
- 使用上下文治理模块完成近重复片段过滤、字符与 Token 双预算、来源标注和敏感信息脱敏。
- 使用 PostgreSQL 保存文档元数据和会话记录，使用 Redis TTL Cache 缓存高频查询；Redis 不可用时自动降级为进程内缓存。
- 使用 LangGraph 构建 `retrieve`、`list`、`chat` 三类问答路由。
- 提供 FastAPI 页面和 API，支持 SSE 模型 Token 流式输出、来源展示和检索片段查看。
- 提供离线导入测试、端到端测试、API 测试和可复现检索评测。
- 提供 Docker Compose 和 GitHub Actions，便于本地部署与持续集成。

## 技术栈

Python、FastAPI、LangChain、LangGraph、Embedding、BM25、RRF、Milvus、PostgreSQL、Redis、Docker Compose、SSE

## 系统流程

```text
企业文档
  -> 文档解析与格式统一
  -> Embedding 语义边界分块 / 递归切分降级
  -> 向量写入 Milvus + 元数据写入 PostgreSQL
  -> Query Rewrite
  -> Milvus 向量召回 + BM25 关键词召回
  -> RRF 融合 + Rerank 精排
  -> 上下文去重、预算控制、来源标注与脱敏
  -> 大模型生成带来源回答
  -> FastAPI / SSE / Redis 查询缓存
```

## 目录结构

```text
smartkb-rag/
├── app.py                  # FastAPI 应用、页面、API 与 SSE
├── cache_store.py          # Redis / 进程内 TTL 缓存
├── config.py               # 环境变量与运行参数
├── agent/                  # LangGraph RAG 路由
├── database/               # Milvus 与 PostgreSQL 访问层
├── eval/                   # 路由和检索评测
├── models/                 # LLM 与 Embedding 适配层
├── rag/                    # 加载、分块、检索、上下文与生成链
├── docs/                   # 架构、部署和评测说明
├── test_imports.py         # 模块与关键能力离线检查
├── test_e2e.py             # RAG 主链路端到端测试
├── test_api.py             # FastAPI、缓存、上传与 SSE 测试
├── Dockerfile
└── compose.yml
```

## 快速开始

### 本地运行

```bash
git clone <你的仓库地址>
cd smartkb-rag
python -m venv .venv
```

激活虚拟环境后安装依赖：

```bash
python -m pip install -r requirements.txt
```

复制环境变量模板并填写自己的配置：

```bash
cp .env.example .env
```

启动服务：

```bash
uvicorn app:app --host 0.0.0.0 --port 8501
```

浏览器访问 `http://127.0.0.1:8501`，接口文档位于 `http://127.0.0.1:8501/docs`。

### Docker Compose

配置 `.env` 后执行：

```bash
docker compose up --build
```

Compose 会启动 SmartKB、PostgreSQL、Redis、Milvus、etcd 和 MinIO，并配置健康检查与持久卷。

## 关键配置

```env
DEEPSEEK_MODEL=deepseek-v4-flash
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SPLITTER_STRATEGY=semantic
SPLITTER_SEMANTIC_EMBEDDINGS=true
SPLITTER_SEMANTIC_THRESHOLD=0.55
RERANK_MODE=lexical
RAG_CONTEXT_MAX_CHARS=4000
RAG_CONTEXT_MAX_TOKENS=1400
REDIS_URL=redis://127.0.0.1:6379/0
QUERY_CACHE_TTL_SECONDS=600
```

完整配置见 `.env.example`。密钥只应保存在本地 `.env` 中，该文件已被 Git 忽略。

## 主要接口

| 接口 | 说明 |
| --- | --- |
| `GET /api/health` | 检查模型、数据库、向量库和缓存状态 |
| `GET /api/documents` | 查看已入库文档 |
| `POST /api/index-text` | 入库文本内容 |
| `POST /api/index-file` | 上传并入库文件 |
| `DELETE /api/documents/{file_name}` | 删除文档、向量并清理缓存 |
| `POST /api/query` | 普通知识库问答 |
| `GET /api/query/stream` | SSE 流式知识库问答 |

## 测试与评测

默认检查不依赖外部 API、Milvus 或 PostgreSQL：

```bash
python test_imports.py
python test_e2e.py
python test_api.py
python eval/agent_eval.py
python eval/retrieval_eval.py
```

配置外部服务后可执行连通性检查：

```bash
python test_imports.py --live
python test_e2e.py --live
```

## 项目文档

- `docs/ARCHITECTURE.md`：系统分层和数据流。
- `docs/DEPLOYMENT.md`：本地运行、容器部署和故障排查。
- `docs/EVALUATION.md`：测试范围与检索指标定义。
- `docs/PROJECT_REPORT.md`：设计取舍和模块职责。
