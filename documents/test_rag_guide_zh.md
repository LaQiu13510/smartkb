# RAG技术完全指南

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的AI技术。

## 核心组件

1. **文档加载器**: 支持PDF、DOCX、TXT、Markdown等格式
2. **文本分割器**: 使用递归字符分割算法，chunk_size=500, overlap=50
3. **嵌入模型**: 智谱AI embedding-2，输出1024维向量
4. **向量数据库**: Milvus，使用COSINE相似度和IVF_FLAT索引
5. **混合检索**: 向量语义检索 + BM25关键词检索，通过RRF算法融合

## 混合检索公式

RRF_score(d) = w_vector / (k + rank_vector(d)) + w_bm25 / (k + rank_bm25(d))

推荐参数: w_vector=0.6, w_bm25=0.4, k=60

## 性能优化建议

1. 调整chunk_size: 太小失去上下文，太大会引入噪声
2. 使用Cross-encoder Reranker精排检索结果
3. 实施Query Rewriting提升模糊查询的召回率
4. 部署Redis缓存热点查询结果

## 部署要求

- CPU: 8核以上
- 内存: 32GB以上
- 磁盘: 500GB SSD
- GPU: 可选，用于本地Embedding加速