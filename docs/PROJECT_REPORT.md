# SmartKB 项目设计说明

## 项目目标

SmartKB 用于降低企业内部资料查找成本。系统把文档解析、语义分块、向量化、混合检索、上下文治理和答案生成连接成可运行链路，并通过来源标注提升回答的可追踪性。

## 关键设计

### 文档处理

`DocumentLoader` 统一处理 Markdown、TXT、PDF 和 DOCX。`TextSplitter` 优先利用 Embedding 余弦相似度检测主题变化，同时保留标题和段落边界；模型服务异常时自动回退，避免入库流程完全中断。

### 混合检索

系统同时使用 Milvus 向量检索与 BM25 关键词检索。向量检索适合语义改写，BM25 适合缩写、配置项和专业术语。两路结果经 RRF 融合，再对扩大后的候选集执行精排。

### 上下文管理

检索结果不会直接拼接进入 Prompt。上下文模块先去除重复证据，再根据字符和 Token 预算截断，并为每段内容保留文件名、页码和相关度，同时对密钥和连接串进行脱敏。

### 缓存与一致性

Redis TTL Cache 用于减少高频问题的重复检索和模型调用。文档入库、覆盖更新或删除后，系统同步清理查询缓存并重建 Retriever，避免继续返回旧内容。

### 工程化

FastAPI 同时提供 Web 页面和结构化 API；SSE 直接转发模型输出 Token。Docker Compose 提供完整依赖环境，GitHub Actions 运行编译、Compose 配置校验、离线测试和评测脚本。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `rag/loader.py` | 文件解析和元数据统一 |
| `rag/splitter.py` | Embedding 语义分块与递归降级 |
| `rag/retriever.py` | 向量召回、BM25、RRF、Query Rewrite、Rerank |
| `rag/context.py` | 去重、预算、来源和脱敏 |
| `rag/chain.py` | 提示词构建与模型生成 |
| `cache_store.py` | Redis / 内存缓存统一接口 |
| `database/` | 向量与关系数据访问 |
| `agent/` | LangGraph 问答路由 |
| `app.py` | FastAPI、文档管理、问答与 SSE |
| `eval/` | 路由和检索评测 |

## 已知边界

- 默认轻量 Rerank 不等同于训练后的专业排序模型，可通过配置启用 Cross Encoder。
- PostgreSQL 元数据和 Milvus 向量更新属于跨存储操作，当前通过顺序执行和错误返回控制风险，未实现分布式事务。
- 进程内缓存只适合单实例降级，多实例部署应配置 Redis。
