# SmartKB 系统架构

## 分层设计

```text
接入层
  app.py：FastAPI 页面、文档管理、问答 API、SSE

Agent 层
  agent/graph.py：chat、list、retrieve 路由
  agent/tools.py：知识库工具封装

RAG 层
  rag/loader.py：Markdown、TXT、PDF、DOCX 解析
  rag/splitter.py：Embedding 语义边界、结构边界和递归切分
  rag/retriever.py：向量召回、BM25、RRF、Query Rewrite、Rerank
  rag/context.py：去重、预算、来源和脱敏
  rag/chain.py：提示词构建与模型流式生成

存储层
  database/milvus_store.py：向量和片段载荷
  database/postgres_store.py：文档元数据和会话记录
  cache_store.py：Redis / 内存 TTL 缓存

模型层
  models/embedding.py：Embedding 后端适配
  models/llm.py：DeepSeek 兼容模型适配

质量层
  eval/：路由与检索评测
  test_*.py：导入、端到端和 API 测试
```

## 文档入库流程

1. FastAPI 接收文本或文件，并校验扩展名、文件名和大小。
2. `DocumentLoader` 将不同格式统一为带元数据的文档对象。
3. `TextSplitter` 批量计算相邻语义单元的 Embedding 相似度，并结合标题、段落和长度预算确定边界。
4. Embedding 服务不可用时，分块器自动降级到词汇相似度和递归字符切分。
5. 文档片段向量写入 Milvus，文档元数据写入 PostgreSQL。
6. 入库、更新或删除成功后，系统清理查询缓存并重建 BM25 索引。

## 检索流程

```text
用户问题
  -> 轻量 Query Rewrite
  -> Milvus 向量召回
  -> BM25 关键词召回
  -> RRF 融合扩大候选集
  -> 词汇覆盖度或 Cross Encoder 精排
  -> Top-K 证据
```

向量召回负责语义改写和近义表达，BM25 负责缩写、配置名和专业术语。RRF 使用排名而不是原始分数融合不同检索器，避免两类分数量纲不一致。精排在扩大后的候选集上执行，避免过早截断有效证据。

## 上下文治理

`RAGContextManager` 在生成前执行以下处理：

- 精确重复和近重复片段过滤。
- 字符预算与 Token 预算双重限制。
- 文件名、页码和相关度标注。
- API Key 与数据库连接串脱敏。
- Prompt Injection 基础约束由生成链统一加入系统提示词。

## 缓存一致性

查询缓存键由规范化问题和 `top_k` 共同组成。文档内容发生变化后，通过前缀删除清理相关缓存，同时清除进程内 Retriever 单例，确保 BM25 语料和向量库状态保持一致。

## 降级策略

- Redis 不可用：使用进程内 TTL 缓存。
- Embedding 语义分块不可用：使用词汇边界和递归切分。
- Cross Encoder 不可用：使用轻量词汇精排。
- 单路检索失败：另一检索器仍可返回结果。
