# RAG (检索增强生成) 技术白皮书

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
