"""
RAG 评测模块
============
提供自动化评测功能，用于衡量 RAG 系统的检索和生成质量。

评测指标:
  1. Hit Rate (命中率): 检索结果中是否包含正确答案的比例
  2. MRR (Mean Reciprocal Rank): 第一个相关文档的平均倒数排名
  3. Context Relevance: 检索到的文档与问题的相关性
  4. Latency: 平均响应延迟 (毫秒)
"""

import time
from typing import List

from langchain_core.messages import HumanMessage

from rag.retriever import HybridRetriever
from rag.chain import get_rag_chain
from models.llm import get_llm


class RAGEvaluator:
    """RAG 系统评测器"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.rag_chain = get_rag_chain()
        self.judge_llm = get_llm(temperature=0.0)

    def evaluate_single(
        self,
        question: str,
        expected_keywords: List[str] | None = None,
    ) -> dict:
        """
        对单个问题进行评测

        Args:
            question: 测试问题
            expected_keywords: 期望答案中包含的关键词列表 (用于简单判断)

        Returns:
            评测结果字典
        """
        # 1. 检索评测
        start = time.time()
        retrieval_results = self.retriever.search(question, top_k=5)
        retrieval_latency = (time.time() - start) * 1000

        # 计算 Hit Rate 与 MRR。
        # 若提供关键词，则以“首个包含任一关键词的片段排名”为相关性依据；
        # 若未提供关键词，则有检索结果即视为命中，用于自定义问题的快速评测。
        hit, mrr = self._retrieval_metrics(retrieval_results, expected_keywords)

        # 2. 生成评测
        result = self.rag_chain.answer(
            query=question,
            retrieval_results=retrieval_results,
        )

        # 3. LLM 评判 (用 LLM 判断答案是否合理)
        judge_score = self._judge_answer(question, result["answer"], expected_keywords)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "hit": hit,
            "mrr": round(mrr, 4),
            "retrieval_latency_ms": round(retrieval_latency, 1),
            "generation_latency_ms": result["latency_ms"],
            "total_latency_ms": round(retrieval_latency + result["latency_ms"], 1),
            "judge_score": judge_score,
            "retrieved_count": len(retrieval_results),
        }

    def _retrieval_metrics(
        self,
        retrieval_results: List[dict],
        expected_keywords: List[str] | None = None,
    ) -> tuple[int, float]:
        """计算检索命中与第一个相关结果的倒数排名。"""
        if not retrieval_results:
            return 0, 0.0

        if not expected_keywords:
            return 1, 1.0

        normalized_keywords = [
            keyword.lower()
            for keyword in expected_keywords
            if keyword and keyword.strip()
        ]
        if not normalized_keywords:
            return 1, 1.0

        for rank, item in enumerate(retrieval_results, start=1):
            content = item.get("content", "").lower()
            if any(keyword in content for keyword in normalized_keywords):
                return 1, 1.0 / rank

        return 0, 0.0

    def evaluate_batch(
        self,
        questions: List[dict],
        verbose: bool = True,
    ) -> dict:
        """
        批量评测

        Args:
            questions: [{"question": ..., "keywords": [...]}, ...]
            verbose: 是否打印详细结果

        Returns:
            汇总评测报告
        """
        results = []
        for i, q in enumerate(questions):
            if verbose:
                print(f"\n[{i+1}/{len(questions)}] 评测: {q['question'][:50]}...")

            result = self.evaluate_single(
                question=q["question"],
                expected_keywords=q.get("keywords"),
            )
            results.append(result)

        # 汇总统计
        total = len(results)
        hit_count = sum(r["hit"] for r in results)
        avg_mrr = sum(r["mrr"] for r in results) / total if total > 0 else 0
        avg_latency = sum(r["total_latency_ms"] for r in results) / total if total > 0 else 0
        avg_judge = sum(r["judge_score"] for r in results) / total if total > 0 else 0
        avg_retrieved = sum(r["retrieved_count"] for r in results) / total if total > 0 else 0

        report = {
            "total_questions": total,
            "hit_rate": round(hit_count / total, 4) if total > 0 else 0,
            "avg_mrr": round(avg_mrr, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_judge_score": round(avg_judge, 2),
            "avg_retrieved_count": round(avg_retrieved, 1),
            "details": results,
        }

        return report

    def _judge_answer(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str] | None = None,
    ) -> float:
        """
        使用 LLM 评判答案质量

        返回 0.0 ~ 1.0 的分数
        """
        # 如果提供了关键词，先做简单匹配
        keyword_score = 1.0
        if expected_keywords:
            matched = sum(
                1 for kw in expected_keywords
                if kw.lower() in answer.lower()
            )
            keyword_score = matched / len(expected_keywords) if expected_keywords else 1.0

        # LLM 评判
        judge_prompt = f"""请评价以下问答的质量。

问题: {question}

答案: {answer[:1000]}

请从以下维度评分（每项 0-10 分）:
1. 相关性: 答案是否直接回应了问题
2. 准确性: 答案的内容是否合理可信
3. 完整性: 答案是否足够详细
4. 流畅性: 语言表达是否自然流畅

请只输出一个 JSON 格式的评分:
{{"relevance": X, "accuracy": X, "completeness": X, "fluency": X}}

JSON:"""

        try:
            response = self.judge_llm.chat(
                [HumanMessage(content=judge_prompt)]
            )
            import json
            # 尝试解析 JSON
            scores = json.loads(response.strip())
            avg_llm_score = (
                scores.get("relevance", 7)
                + scores.get("accuracy", 7)
                + scores.get("completeness", 7)
                + scores.get("fluency", 7)
            ) / 40  # 归一化到 0-1
        except Exception:
            avg_llm_score = 0.7  # 默认值

        # 综合分数 (关键词 40% + LLM 评判 60%)
        final_score = keyword_score * 0.4 + avg_llm_score * 0.6
        return round(final_score, 2)


def run_quick_eval(
    retriever: HybridRetriever,
    questions: List[dict],
) -> dict:
    """快速评测入口"""
    evaluator = RAGEvaluator(retriever)
    return evaluator.evaluate_batch(questions)
