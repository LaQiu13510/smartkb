"""
混合检索器
==========
实现向量检索 + BM25 关键词检索的混合召回策略。
使用 RRF (Reciprocal Rank Fusion) 融合两种检索结果。

检索策略演进 (记录在项目报告中):
  版本 1.0: 纯向量检索 (Milvus COSINE) → 关键词匹配能力弱
  版本 2.0: 混合检索 (向量 + BM25 + RRF) → 兼顾语义和关键词
  版本 3.0: Query rewrite + Reranker → 默认轻量词汇精排，可选 CrossEncoder 精排 ← 当前方案
"""

from typing import List, Optional

from rank_bm25 import BM25Okapi
import jieba

from database.milvus_store import MilvusStore
from models.embedding import get_embedding_model


def _tokenize(text: str) -> List[str]:
    """中文分词 (jieba) + 英文小写分词"""
    # jieba 分词
    tokens = list(jieba.cut(text))
    # 过滤空字符和纯空白
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens


class HybridRetriever:
    """
    混合检索器：向量检索 + BM25 关键词检索 → RRF 融合
    """

    def __init__(
        self,
        milvus_store: MilvusStore,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        enable_query_rewrite: bool = True,
        enable_rerank: bool = True,
        rerank_mode: str = "lexical",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            milvus_store: Milvus 向量存储
            top_k: 最终返回结果数
            vector_weight: 向量检索 RRF 权重
            bm25_weight: BM25 检索 RRF 权重
            rrf_k: RRF 平滑参数
        """
        self.milvus = milvus_store
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_rerank = enable_rerank
        self.rerank_mode = rerank_mode
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None

        # BM25 索引 (需要构建)
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[dict] = []  # 用于映射 BM25 结果回原始文档

    # ---- BM25 索引管理 ----

    def build_bm25_index(self, documents: List[dict]):
        """
        从文档列表构建 BM25 索引

        Args:
            documents: [{"chunk_id": ..., "content": ..., ...}, ...]
        """
        if not documents:
            return

        tokenized = [_tokenize(doc["content"]) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = documents
        print(f"[Retriever] BM25 索引构建完成: {len(documents)} 篇文档")

    # ---- 检索 ----

    def search(
        self,
        query: str,
        top_k: int | None = None,
        return_scores: bool = True,
    ) -> List[dict]:
        """
        混合检索入口

        Args:
            query: 查询文本
            top_k: 返回结果数 (默认使用实例配置)
            return_scores: 是否在结果中包含融合分数

        Returns:
            检索结果列表，按 RRF 融合分数降序排列
        """
        if top_k is None:
            top_k = self.top_k

        rewritten_query = self._rewrite_query(query) if self.enable_query_rewrite else query

        # 1. 向量检索
        vector_results = self._vector_search(rewritten_query, top_k * 2)

        # 2. BM25 关键词检索
        bm25_results = self._bm25_search(rewritten_query, top_k * 2)

        # 3. RRF 融合
        fused = self._rrf_fusion(vector_results, bm25_results, top_k)
        if self.enable_rerank:
            fused = self._rerank_results(query, fused)[:top_k]

        if not return_scores:
            for result in fused:
                result.pop("hybrid_score", None)
                result.pop("vector_score", None)
                result.pop("bm25_score", None)
                result.pop("rerank_score", None)
                result.pop("cross_encoder_score", None)

        return fused

    def search_vector_only(self, query: str, top_k: int | None = None) -> List[dict]:
        """只使用向量检索，便于评测对比。"""
        return self._vector_search(query, top_k or self.top_k)

    def search_bm25_only(self, query: str, top_k: int | None = None) -> List[dict]:
        """只使用 BM25 检索，便于评测对比。"""
        return self._bm25_search(query, top_k or self.top_k)

    def _vector_search(self, query: str, top_k: int) -> List[dict]:
        """向量检索 (Milvus)"""
        try:
            embedding_model = get_embedding_model()
            query_vec = embedding_model.embed_query(query)
            results = self.milvus.search(query_vec, top_k=top_k)
            for r in results:
                r["source"] = "vector"
            return results
        except Exception as e:
            print(f"[Retriever] 向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """BM25 关键词检索"""
        if self._bm25 is None:
            return []

        try:
            tokenized_query = _tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)

            # 取 Top-K
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_indices = indexed[:top_k]

            results = []
            for idx, score in top_indices:
                if score > 0:
                    doc = self._bm25_docs[idx]
                    results.append({
                        "chunk_id": doc.get("chunk_id", ""),
                        "content": doc.get("content", ""),
                        "file_name": doc.get("file_name", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "source_page": doc.get("source_page", 0),
                        "score": float(score),
                        "source": "bm25",
                    })
            return results
        except Exception as e:
            print(f"[Retriever] BM25 检索失败: {e}")
            return []

    def _rewrite_query(self, query: str) -> str:
        """轻量查询扩展，避免引入额外 LLM 调用。"""
        expansions = {
            "rag": "检索增强生成 retrieval augmented generation",
            "混合检索": "hybrid search bm25 向量检索 rrf",
            "rrf": "reciprocal rank fusion 融合排序",
            "部署": "环境 配置 数据库 milvus postgresql",
            "架构": "模块 流程 组件 architecture",
        }
        lower = query.lower()
        extra = []
        for key, value in expansions.items():
            if key in lower and value not in lower:
                extra.append(value)
        return query if not extra else f"{query} {' '.join(extra)}"

    def _rerank_results(self, query: str, results: List[dict]) -> List[dict]:
        """精排：优先 CrossEncoder，可失败回退到轻量词汇精排。"""
        if self.rerank_mode == "cross_encoder":
            ranked = self._cross_encoder_rerank(query, results)
            if ranked:
                return ranked
        return self._lexical_rerank(query, results)

    def _lexical_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """基于查询词覆盖度做轻量精排。"""
        query_tokens = set(_tokenize(query))
        for result in results:
            content_tokens = set(_tokenize(result.get("content", "")))
            overlap = len(query_tokens & content_tokens)
            base_score = float(result.get("hybrid_score", result.get("score", 0.0)))
            result["rerank_score"] = round(base_score + min(overlap * 0.0005, 0.003), 6)
        return sorted(results, key=lambda item: item.get("rerank_score", 0), reverse=True)

    def _cross_encoder_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """使用 CrossEncoder 对候选片段精排；不可用时返回空列表。"""
        if not results:
            return []
        try:
            model = self._get_cross_encoder()
            pairs = [(query, item.get("content", "")) for item in results]
            scores = model.predict(pairs)
            for item, score in zip(results, scores):
                item["cross_encoder_score"] = float(score)
                item["rerank_score"] = float(score)
            return sorted(results, key=lambda item: item.get("cross_encoder_score", 0), reverse=True)
        except Exception as exc:
            print(f"[Retriever] CrossEncoder 精排不可用，回退词汇精排: {str(exc)[:120]}")
            return []

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def _rrf_fusion(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法

        公式: RRF_score(d) = Σ w_i / (k + rank_i(d))
          其中 w_i 是各检索器的权重, k 是平滑参数
        """
        # 构建 chunk_id → 结果的映射
        merged: dict[str, dict] = {}

        # 向量检索结果
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**result, "vector_rank": rank, "vector_score": result["score"]}
            else:
                merged[chunk_id].update({
                    "vector_rank": rank,
                    "vector_score": result["score"],
                })

        # BM25 检索结果
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {**result, "bm25_rank": rank, "bm25_score": result["score"]}
            else:
                merged[chunk_id].update({
                    "bm25_rank": rank,
                    "bm25_score": result["score"],
                })

        # 计算 RRF 分数
        for chunk_id, data in merged.items():
            rrf_score = 0.0
            if "vector_rank" in data:
                rrf_score += self.vector_weight / (self.rrf_k + data["vector_rank"])
            if "bm25_rank" in data:
                rrf_score += self.bm25_weight / (self.rrf_k + data["bm25_rank"])
            data["hybrid_score"] = round(rrf_score, 6)

        # 按 RRF 分数降序排列
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return sorted_results[:top_k]

    @property
    def bm25_ready(self) -> bool:
        """BM25 索引是否就绪"""
        return self._bm25 is not None and len(self._bm25_docs) > 0


# 工厂函数
def create_retriever(
    milvus_store: MilvusStore,
    top_k: int = 5,
) -> HybridRetriever:
    return HybridRetriever(
        milvus_store=milvus_store,
        top_k=top_k,
    )
